"""Evaluation: HellaSwag-style retention + attention-output cosine.

Paper metrics:
1. HellaSwag retention = (transfer accuracy) / (standalone accuracy) * 100%
   Paper Tier 1: 73-98%, Tier 2: 42-44%
2. Attention-output cosine: cosine between attention output from mapped KV vs ground-truth KV
   (paper: r=0.57 with HellaSwag retention, better than R^2 r=-0.20)
"""
import torch
import torch.nn.functional as F
from kv_transfer.kv_cache import (
    extract_kv_cache, strip_kv_rope, get_model_config,
)
from kv_transfer.mapper import apply_mapper
from kv_transfer.rope import compute_rope_cos_sin
from kv_transfer.config import SOURCE_DEVICE

# 50 HellaSwag-style questions: context + 4 choices + correct label
# Balanced: ~12-13 correct per position
QUESTIONS = [
    {"context": "A woman is seen walking down a street. She approaches a building and", "choices": [" enters through the front door.", " flies away into the sky.", " starts breakdancing.", " turns into a bird."], "label": 0},
    {"context": "A man is cooking in a kitchen. He adds vegetables to the pan and", "choices": [" stirs them with a spatula.", " throws the pan out the window.", " begins to juggle the vegetables.", " falls asleep on the floor."], "label": 0},
    {"context": "The student opened the textbook and began to read. After a few minutes, she", "choices": [" closed the book and went to sleep.", " took detailed notes on the chapter.", " ate the textbook pages.", " started singing loudly."], "label": 1},
    {"context": "A car is driving on a highway. The driver sees a red light ahead and", "choices": [" accelerates to run through it.", " applies the brakes to stop.", " jumps out of the car.", " starts flying the car."], "label": 1},
    {"context": "The programmer typed code into the editor. After running the program, she found a bug and", "choices": [" submitted it as a feature.", " debugged and fixed the issue.", " threw the computer away.", " started crying."], "label": 1},
    {"context": "A chef is preparing a salad. She washes the lettuce, chops the tomatoes, and then", "choices": [" tosses everything in a bowl.", " sets the kitchen on fire.", " feeds the vegetables to the dog.", " begins to paint with the dressing."], "label": 0},
    {"context": "The scientist conducted an experiment. She mixed two chemicals together and", "choices": [" drank the mixture immediately.", " observed the reaction carefully.", " poured it on her hands.", " threw the beaker at the wall."], "label": 1},
    {"context": "A musician picked up a guitar. She tuned the strings and then", "choices": [" smashed it on the stage.", " began to play a melody.", " ate the guitar strings.", " used it as a pillow."], "label": 1},
    {"context": "The teacher entered the classroom. She wrote the date on the board and", "choices": [" asked the students to open their books.", " jumped on the desk and screamed.", " threw chalk at the students.", " left immediately."], "label": 0},
    {"context": "A construction worker is operating a crane. He needs to lift steel beams to the top floor, so he", "choices": [" uses the crane controls to lift them.", " carries them on his back.", " asks a bird to fly them up.", " throws them by hand."], "label": 0},
    {"context": "The gardener planted seeds in the soil. She watered them and", "choices": [" poured gasoline on them.", " waited for them to grow.", " ate all the seeds.", " covered them with concrete."], "label": 1},
    {"context": "A doctor examined the patient. She checked the vital signs and then", "choices": [" prescribed appropriate medication.", " told the patient to run a marathon.", " gave the patient a haircut.", " started dancing in the room."], "label": 0},
    {"context": "The athlete stretched before the race. When the starting gun fired, she", "choices": [" sat down and cried.", " sprinted forward with all her strength.", " walked backwards.", " fell asleep on the track."], "label": 1},
    {"context": "A painter set up her easel. She mixed colors on her palette and then", "choices": [" began to paint on the canvas.", " ate the paint.", " threw the palette in the river.", " used the brush as a toothpick."], "label": 0},
    {"context": "The librarian shelved books. She sorted them by genre and", "choices": [" burned them in the fireplace.", " placed them in alphabetical order.", " threw them out the window.", " used them as building blocks."], "label": 1},
    {"context": "A fisherman cast his line into the lake. He waited patiently and then", "choices": [" felt a tug on the line.", " dove into the water headfirst.", " threw his rod away.", " fell asleep and snored."], "label": 0},
    {"context": "The mechanic opened the car hood. She inspected the engine and", "choices": [" replaced the faulty spark plugs.", " poured milk into the engine.", " painted the engine pink.", " started singing to the car."], "label": 0},
    {"context": "A photographer raised her camera. She adjusted the focus and", "choices": [" ate the camera lens.", " pressed the shutter button.", " threw the camera in the lake.", " used it as a mirror."], "label": 1},
    {"context": "The barber picked up scissors. She combed the client's hair and", "choices": [" began to cut it evenly.", " shaved off all the client's eyebrows.", " used the scissors to cut the chair.", " started juggling the scissors."], "label": 0},
    {"context": "A pilot entered the cockpit. She checked the instruments and", "choices": [" jumped out the window.", " started the engine for takeoff.", " served drinks to passengers.", " fell asleep in the seat."], "label": 1},
    {"context": "The farmer harvested crops. She loaded them onto a truck and", "choices": [" drove them to the market.", " set them on fire.", " buried them in the field.", " fed them to the tractor."], "label": 0},
    {"context": "A swimmer dove into the pool. She reached the other end and", "choices": [" turned around and swam back.", " sank to the bottom and stayed.", " flew out of the pool.", " started reading a book underwater."], "label": 0},
    {"context": "The electrician checked the wiring. She found a short circuit and", "choices": [" touched the bare wire with wet hands.", " repaired it with proper tools.", " poured water on the wires.", " ignored it and went home."], "label": 1},
    {"context": "A baker kneaded dough. She shaped it into a loaf and", "choices": [" threw it at the wall.", " placed it in the oven to bake.", " used it as a pillow.", " fed it to the cat."], "label": 1},
    {"context": "The detective examined the crime scene. She collected evidence and", "choices": [" ate the evidence.", " documented everything carefully.", " danced on the evidence.", " threw it out the window."], "label": 1},
    {"context": "A tailor measured the fabric. She cut it with scissors and", "choices": [" began to sew the pieces together.", " ate the fabric.", " wrapped herself in it like a mummy.", " burned it with a lighter."], "label": 0},
    {"context": "The journalist opened her laptop. She typed furiously and", "choices": [" smashed the keyboard with a hammer.", " finished the article before deadline.", " threw the laptop in the pool.", " used it as a cutting board."], "label": 1},
    {"context": "A surgeon scrubbed her hands. She put on gloves and", "choices": [" entered the operating room.", " started juggling scalpels.", " went to sleep on the table.", " painted her mask red."], "label": 0},
    {"context": "The taxi driver picked up a passenger. She checked the GPS and", "choices": [" drove to the requested destination.", " drove into a lake.", " fell asleep at the wheel.", " started driving backwards."], "label": 0},
    {"context": "A child built a sandcastle. She patted the sand and", "choices": [" decorated it with shells.", " kicked it down immediately.", " ate the wet sand.", " poured gasoline on it."], "label": 0},
    {"context": "The plumber crawled under the sink. She tightened a pipe and", "choices": [" bit the pipe with her teeth.", " checked for leaks.", " poured acid down the drain.", " fell asleep under the sink."], "label": 1},
    {"context": "A DJ put on headphones. She adjusted the mixer and", "choices": [" started playing music for the crowd.", " threw the headphones in the crowd.", " ate the vinyl records.", " fell asleep on the turntable."], "label": 0},
    {"context": "The researcher analyzed data. She ran statistical tests and", "choices": [" ate the computer monitor.", " published the findings in a journal.", " deleted all the data.", " used the keyboard as a pillow."], "label": 1},
    {"context": "A carpenter measured the wood. She marked a line and", "choices": [" sawed along the line carefully.", " threw the saw at the wall.", " ate the sawdust.", " used the wood as a surfboard."], "label": 0},
    {"context": "The nurse checked the IV drip. She adjusted the flow rate and", "choices": [" disconnected it and drank the medicine.", " monitored the patient's condition.", " poured it on the floor.", " used the IV stand as a microphone."], "label": 1},
    {"context": "A soccer player dribbled past a defender. She approached the goal and", "choices": [" kicked the ball into the net.", " picked up the ball and ran.", " sat on the ball and bounced.", " threw the ball at the referee."], "label": 0},
    {"context": "The architect drew blueprints. She calculated dimensions and", "choices": [" ate the blueprint paper.", " finalized the building design.", " used the pencil as a drumstick.", " drew stick figures instead."], "label": 1},
    {"context": "A firefighter entered the burning building. She carried a hose and", "choices": [" sprayed water on the flames.", " added more fire to the building.", " fell asleep in the flames.", " started cooking marshmallows."], "label": 0},
    {"context": "The cashier scanned items. She totaled the price and", "choices": [" threw the money in the air.", " processed the payment.", " ate the receipts.", " gave the customer a haircut."], "label": 1},
    {"context": "A veterinarian examined a dog. She checked its heartbeat and", "choices": [" administered the necessary vaccine.", " taught the dog to drive.", " painted the dog blue.", " gave the dog a haircut."], "label": 0},
    {"context": "The judge entered the courtroom. She sat down and", "choices": [" began the proceedings.", " did a backflip over the bench.", " fell asleep in the chair.", " started singing opera."], "label": 0},
    {"context": "A journalist interviewed a politician. She asked a tough question and", "choices": [" threw the microphone at him.", " listened carefully to the response.", " fell asleep during the answer.", " started dancing in the room."], "label": 1},
    {"context": "The astronaut floated in space. She checked her oxygen supply and", "choices": [" took off her helmet to breathe.", " continued the spacewalk.", " threw her tools into the void.", " fell asleep floating."], "label": 1},
    {"context": "A florist arranged flowers. She trimmed the stems and", "choices": [" placed them in a vase with water.", " ate the flower stems.", " threw them in the trash.", " used them as drumsticks."], "label": 0},
    {"context": "The coach blew the whistle. The players stopped and", "choices": [" gathered around for instructions.", " ran away from the field.", " lay down and fell asleep.", " started a food fight."], "label": 0},
    {"context": "A mechanic checked the tire pressure. She found it low and", "choices": [" inflated the tire with air.", " deflated all the other tires.", " ate the valve caps.", " painted the tires gold."], "label": 0},
    {"context": "The waiter carried plates. She balanced them on her arm and", "choices": [" dropped them on the floor.", " served them to the customers.", " threw them at the wall.", " ate all the food."], "label": 1},
    {"context": "A scientist looked through a microscope. She adjusted the lens and", "choices": [" observed the cell structure.", " ate the glass slide.", " used the microscope as a pillow.", " threw it across the lab."], "label": 0},
    {"context": "The translator read a document. She understood the source language and", "choices": [" ate the document.", " produced an accurate translation.", " drew pictures on it.", " fell asleep on the pages."], "label": 1},
    {"context": "A blacksmith heated iron. She hammered it on the anvil and", "choices": [" shaped it into a horseshoe.", " threw the hot iron at the wall.", " ate the hot metal.", " used the hammer as a toothbrush."], "label": 0},
]

# Prompts for attention-output cosine evaluation
TEST_PROMPTS = [
    "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Brazil is",
    "The transformer architecture revolutionized natural language processing by",
    "In machine learning, gradient descent is an optimization algorithm used to",
    "Climate change refers to long-term shifts in global temperatures and",
    "Python is a high-level programming language known for its simplicity and",
]


def hellaswag_loglikelihood(model, tokenizer, context, choice, device, mapped_kv=None, num_layers=None):
    """Compute log-likelihood of choice given context.

    If mapped_kv is provided, use mapped KV cache. Otherwise use standalone.
    """
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    choice_ids = tokenizer(choice, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        if mapped_kv is not None:
            # Transfer: use mapped KV
            out = model(ctx_ids, use_cache=True)
            cache = out.past_key_values
            for l in range(num_layers):
                dev = cache.layers[l].keys.device
                cache.layers[l].keys = mapped_kv[l]["keys"].to(dev)
                cache.layers[l].values = mapped_kv[l]["values"].to(dev)
            # Forward choice tokens with mapped cache
            choice_dev = choice_ids.to(cache.layers[0].keys.device)
            out = model(choice_dev, past_key_values=cache, use_cache=False)
            logits = out.logits  # [1, choice_len, vocab]
            # Log-likelihood of choice tokens: sum log P(choice_token[i] | context + choice[:i])
            # logits[i] predicts choice[i+1], so compare logits[:-1] with choice[1:]
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = choice_dev[:, 1:]
            ll = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1).sum().item()
        else:
            # Standalone: 14B prefill context, then forward choice
            out = model(ctx_ids, use_cache=True)
            cache = out.past_key_values
            choice_dev = choice_ids.to(device)
            out = model(choice_dev, past_key_values=cache, use_cache=False)
            logits = out.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = choice_dev[:, 1:]
            ll = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1).sum().item()

    return ll


def compute_attention_output_cosine(model, ids, mapped_kv, num_layers, device):
    """Compute attention-output cosine between mapped KV and ground-truth KV.

    Uses forward hooks to capture attention outputs from each layer.
    Compares the last token's attention output.
    """
    # Hook to capture attention outputs
    std_attn = {}
    tr_attn = {}

    def make_hook(storage, key):
        def hook(module, input, output):
            # output is (attn_output, ...) tuple
            if isinstance(output, tuple):
                storage[key] = output[0].detach()
            else:
                storage[key] = output.detach()
        return hook

    # Register hooks on self_attn for each layer
    hooks = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        h = layer.self_attn.register_forward_hook(make_hook(std_attn, layer_idx))
        hooks.append(h)

    # Standalone forward (ground truth)
    ids_dev = ids.to(device)
    with torch.no_grad():
        model(ids_dev, use_cache=False)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Register hooks for transfer forward
    hooks = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        h = layer.self_attn.register_forward_hook(make_hook(tr_attn, layer_idx))
        hooks.append(h)

    # Transfer forward: 14B forward to init cache, replace, then forward last token
    with torch.no_grad():
        out = model(ids_dev, use_cache=True)
        cache = out.past_key_values
        for l in range(num_layers):
            dev = cache.layers[l].keys.device
            cache.layers[l].keys = mapped_kv[l]["keys"].to(dev)
            cache.layers[l].values = mapped_kv[l]["values"].to(dev)
        last_token = ids_dev[:, -1:].to(cache.layers[0].keys.device)
        model(last_token, past_key_values=cache, use_cache=False)

    for h in hooks:
        h.remove()

    # Compute cosine per layer (last token position)
    cosines = []
    for layer_idx in range(num_layers):
        if layer_idx in std_attn and layer_idx in tr_attn:
            std_vec = std_attn[layer_idx][0, -1:, :].flatten().float().cpu()
            tr_vec = tr_attn[layer_idx][0, -1:, :].flatten().float().cpu()
            cos = F.cosine_similarity(std_vec.unsqueeze(0), tr_vec.unsqueeze(0)).item()
            cosines.append(cos)

    return cosines


def run_attention_cosine_eval(model_8b, model_14b, tok, mapper):
    """Run attention-output cosine evaluation on TEST_PROMPTS."""
    rt8, hd8, nl8, nk8 = get_model_config(model_8b)
    rt14, hd14, nl14, nk14 = get_model_config(model_14b)
    tdev = next(model_14b.parameters()).device

    print("=" * 60, flush=True)
    print("1. Attention-Output Cosine (paper's key predictor, r=0.57)", flush=True)
    print("=" * 60, flush=True)

    all_cosines = []
    for i, prompt in enumerate(TEST_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to(SOURCE_DEVICE)
        ids_14b = ids.to(tdev)

        with torch.no_grad():
            # 8B prefill + map
            source_kv = extract_kv_cache(model_8b, ids)
            kv_stripped = strip_kv_rope(source_kv, rt8, hd8)
            seq_len = kv_stripped[0]["keys"].shape[2]
            cos, sin = compute_rope_cos_sin(hd14, rt14, seq_len, SOURCE_DEVICE)
            mapped_kv = apply_mapper(kv_stripped, mapper, nl14, nk14, target_rope_cos=cos, target_rope_sin=sin)

        # Compute attention-output cosine
        cosines = compute_attention_output_cosine(model_14b, ids_14b, mapped_kv, nl14, tdev)
        avg_cos = sum(cosines) / len(cosines)
        all_cosines.append(avg_cos)
        print(f"  P{i+1}: attn-out cos = {avg_cos:.4f} (min={min(cosines):.4f}, max={max(cosines):.4f})", flush=True)

    overall_avg = sum(all_cosines) / len(all_cosines)
    print(f"\n  Overall attention-output cosine: {overall_avg:.4f}", flush=True)
    print(f"  (paper: r=0.57 with HellaSwag retention across 12 pairs)", flush=True)


def run_hellaswag_eval(model_8b, model_14b, tok, mapper):
    """Run 50-question HellaSwag-style retention evaluation."""
    rt8, hd8, nl8, nk8 = get_model_config(model_8b)
    rt14, hd14, nl14, nk14 = get_model_config(model_14b)
    tdev = next(model_14b.parameters()).device

    total = len(QUESTIONS)
    std_correct = 0
    tr_correct = 0
    both_correct = 0
    std_only = 0
    tr_only = 0
    both_wrong = 0

    print(f"\n{'='*60}", flush=True)
    print("2. HellaSwag-style Retention", flush=True)
    print("=" * 60, flush=True)
    print(f"Evaluating {total} HellaSwag-style questions...\n", flush=True)

    for i, item in enumerate(QUESTIONS):
        context = item["context"]
        choices = item["choices"]
        label = item["label"]

        # Standalone
        std_lls = [hellaswag_loglikelihood(model_14b, tok, context, c, tdev) for c in choices]
        std_pred = max(range(4), key=lambda x: std_lls[x])

        # Transfer
        with torch.no_grad():
            ctx_ids = tok(context, return_tensors="pt").input_ids.to(SOURCE_DEVICE)
            source_kv = extract_kv_cache(model_8b, ctx_ids)
            kv_stripped = strip_kv_rope(source_kv, rt8, hd8)
            seq_len = kv_stripped[0]["keys"].shape[2]
            cos, sin = compute_rope_cos_sin(hd14, rt14, seq_len, SOURCE_DEVICE)
            mapped_kv = apply_mapper(kv_stripped, mapper, nl14, nk14, target_rope_cos=cos, target_rope_sin=sin)

        tr_lls = [hellaswag_loglikelihood(model_14b, tok, context, c, tdev, mapped_kv, nl14) for c in choices]
        tr_pred = max(range(4), key=lambda x: tr_lls[x])

        std_ok = std_pred == label
        tr_ok = tr_pred == label

        if std_ok and tr_ok: both_correct += 1
        elif std_ok and not tr_ok: std_only += 1
        elif not std_ok and tr_ok: tr_only += 1
        else: both_wrong += 1

        std_correct += std_ok
        tr_correct += tr_ok

        status = ""
        if std_ok and tr_ok: status = "both \u2713"
        elif std_ok and not tr_ok: status = "std only \u2713"
        elif not std_ok and tr_ok: status = "tr only \u2713"
        else: status = "both \u2717"

        if (i+1) % 10 == 0 or not std_ok or not tr_ok:
            print(f"  Q{i+1:2d}: {status} | std={std_pred} tr={tr_pred} label={label}", flush=True)

    std_acc = std_correct / total * 100
    tr_acc = tr_correct / total * 100
    retention = tr_acc / std_acc * 100 if std_acc > 0 else 0

    print(f"\n{'='*60}", flush=True)
    print(f"HellaSwag-style Results ({total} questions)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Standalone accuracy: {std_correct}/{total} = {std_acc:.1f}%", flush=True)
    print(f"  Transfer accuracy:   {tr_correct}/{total} = {tr_acc:.1f}%", flush=True)
    print(f"  Retention: {retention:.1f}%", flush=True)
    print(f"\n  Breakdown:", flush=True)
    print(f"    Both correct:  {both_correct}", flush=True)
    print(f"    Std only:      {std_only}", flush=True)
    print(f"    Tr only:       {tr_only}", flush=True)
    print(f"    Both wrong:    {both_wrong}", flush=True)
    print(f"\n  Paper: Tier 1 = 73-98%, Tier 2 = 42-44%", flush=True)
