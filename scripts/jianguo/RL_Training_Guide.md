# Complete Guide: RL Training for Math Reasoning

> A beginner-friendly guide to understanding GRPO training in the `miles` codebase.

## Table of Contents

1. [Overview](#part-1-overview)
2. [Training Pipeline](#part-2-training-pipeline)
3. [Reward Computation](#part-3-reward-computation)
4. [GRPO Normalization](#part-4-grpo-normalization)
5. [Loss Computation](#part-5-loss-computation)
6. [Training Entry Point](#part-6-training-entry-point)
7. [Hyperparameters](#part-7-hyperparameters)
8. [Why Things Work](#part-8-why-things-work)
9. [Advanced: Credit Assignment](#part-9-advanced-credit-assignment)
10. [Code Reference](#part-10-code-reference)

---

## Part 1: Overview

### What is GRPO?

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm designed for training language models. Unlike traditional RL that uses a learned value function, GRPO:

- Generates **multiple responses** per prompt
- Computes rewards for each response
- Normalizes rewards **within each group** to get advantages
- Uses these advantages to update the policy

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRPO TRAINING LOOP                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. ROLLOUT PHASE (GPU Inference)                                         │
│      ┌────────────────────────────────────────┐                            │
│      │ Prompt: "Solve 2x + 3 = 7"             │                            │
│      │         ↓                              │                            │
│      │ Generate 8 responses (n_samples=8)     │                            │
│      │         ↓                              │                            │
│      │ Response 0: "<think>...</think>\boxed{2}"  ← correct               │
│      │ Response 1: "<think>...</think>\boxed{5}"  ← wrong                 │
│      │ Response 2: "<think>...</think>\boxed{2}"  ← correct               │
│      │ ...                                    │                            │
│      └────────────────────────────────────────┘                            │
│                                                                             │
│   2. REWARD PHASE (CPU)                                                    │
│      ┌────────────────────────────────────────┐                            │
│      │ Extract answer from \boxed{}           │                            │
│      │ Compare with ground truth              │                            │
│      │ Reward = 1 (correct) or 0 (wrong)      │                            │
│      └────────────────────────────────────────┘                            │
│                                                                             │
│   3. NORMALIZATION PHASE                                                   │
│      ┌────────────────────────────────────────┐                            │
│      │ Within each prompt group:              │                            │
│      │   advantage = reward - mean(rewards)   │                            │
│      │                                        │                            │
│      │ Example: rewards = [1,0,1,1,1,0,1,1]   │                            │
│      │          mean = 0.75                   │                            │
│      │          advantages = [+0.25, -0.75,   │                            │
│      │                        +0.25, +0.25,   │                            │
│      │                        +0.25, -0.75,   │                            │
│      │                        +0.25, +0.25]   │                            │
│      └────────────────────────────────────────┘                            │
│                                                                             │
│   4. TRAINING PHASE (GPU)                                                  │
│      ┌────────────────────────────────────────┐                            │
│      │ For each token in each response:       │                            │
│      │   pg_loss = -clip(ratio) × advantage   │                            │
│      │                                        │                            │
│      │ Positive advantage → increase P(token) │                            │
│      │ Negative advantage → decrease P(token) │                            │
│      └────────────────────────────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Training Pipeline

### File: `miles/ray/rollout.py`

This orchestrates the rollout process:

```python
class RolloutManager:
    async def run_rollout(self, samples):
        # 1. Generate responses using sglang engine
        responses = await self.engine.generate(samples)
        
        # 2. Compute rewards
        for sample in samples:
            reward = await rule_based_rm(self.args, sample)
            sample.reward = reward
        
        # 3. Normalize rewards (GRPO)
        raw_rewards, normalized_rewards = self._reward_post_process(samples)
        
        return samples
```

### File: `miles/backends/megatron_utils/actor.py`

The training actor:

```python
class MegatronTrainRayActor:
    def train(self, rollout_id, rollout_data_ref):
        # 1. Reload process groups if using memory offload
        if self.args.offload_train:
            self.wake_up()
        
        # 2. Get rollout data
        rollout_data = self._get_rollout_data(rollout_data_ref)
        
        # 3. Train
        return self.train_actor(rollout_id, rollout_data)
```

---

## Part 3: Reward Computation

### 3.1 The Reward Function

**File:** `miles/rollout/rm_hub/deepscaler.py`

```python
def get_deepscaler_rule_based_reward(response, label):
    """
    Compute reward for a math response.
    
    Args:
        response: Model's full response including <think>...</think>
        label: Ground truth answer
    
    Returns:
        1 if correct, 0 if wrong
    """
    # Step 1: Extract the answer part (after </think>)
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    else:
        return 0  # No proper format
    
    # Step 2: Extract answer from \boxed{}
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0
    
    # Step 3: Process ground truth
    assert isinstance(label, (str, float, int))
    ground_truths = [label]
    
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    
    if not processed_ground_truths:
        return 0
    
    # Step 4: Grade the answer
    for ground_truth in processed_ground_truths:
        is_correct = (
            grade_answer_mathd(model_answer, ground_truth) or 
            grade_answer_sympy(model_answer, ground_truth)
        )
        if is_correct:
            return 1
    
    return 0
```

### 3.2 Answer Grading

**File:** `miles/rollout/rm_hub/math_utils.py`

```python
def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    """
    String-based grading with normalization.
    
    Handles: "1/2" == "0.5", "\\frac{1}{2}" == "0.5", etc.
    """
    ground_truth_normalized = mathd_normalize_answer(ground_truth)
    given_answer_normalized = mathd_normalize_answer(given_answer)
    
    return ground_truth_normalized == given_answer_normalized


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    """
    Mathematical equivalence via SymPy.
    
    Handles: "2x + 4" == "2(x + 2)", symbolic expressions
    """
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    
    # Check if difference simplifies to 0
    expr = f"({ground_truth_normalized})-({given_normalized})"
    sympy_diff = _sympy_parse(expr)
    simplified = sympy.simplify(sympy_diff)
    
    return simplified == 0
```

### 3.3 Example: Reward Computation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL RESPONSE                                                              │
│ ─────────────                                                               │
│ "<think>                                                                    │
│  Let me solve 2x + 3 = 7                                                   │
│  Subtract 3: 2x = 4                                                        │
│  Divide by 2: x = 2                                                        │
│  </think>                                                                   │
│  The answer is \boxed{2}"                                                   │
│                                                                             │
│ REWARD COMPUTATION                                                          │
│ ──────────────────                                                          │
│ 1. Extract answer after </think>: "The answer is \boxed{2}"                │
│ 2. Extract from \boxed{}: "2"                                              │
│ 3. Compare with ground truth: "2" == "2" ✓                                 │
│ 4. Reward = 1 (correct)                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: GRPO Normalization

### 4.1 Why Normalize?

**Problem:** Raw rewards are just 0 or 1 - they don't tell us which responses are *relatively* better.

**Solution:** Normalize within each prompt group to get advantages.

```
EXAMPLE - 8 responses to same question:

Raw rewards:     [1, 0, 1, 1, 1, 0, 1, 1]  (6 correct, 2 wrong)
Mean:            0.75

Advantages = rewards - mean:
             [+0.25, -0.75, +0.25, +0.25, +0.25, -0.75, +0.25, +0.25]
                ↑ better      ↑ worse
                than avg      than avg
```

### 4.2 Code: GRPO Normalization

**File:** `miles/ray/rollout.py`

```python
def _reward_post_process(self, samples):
    """
    Normalize rewards within each prompt group (GRPO).
    """
    raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
    
    if self.args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        
        # Reshape: [batch, n_samples_per_prompt]
        # Example: [32, 8] for 32 prompts, 8 samples each
        rewards = rewards.view(-1, self.args.n_samples_per_prompt)
        
        # Compute mean within each group
        mean = rewards.mean(dim=-1, keepdim=True)  # [32, 1]
        
        # Subtract mean: now zero-mean within each group
        rewards = rewards - mean
        
        # Optional: divide by std for unit variance
        if self.args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)
        
        return raw_rewards, rewards.flatten().tolist()
```

### 4.3 Understanding Advantages

```
ADVANTAGE = "How much better/worse than average was this response?"

Group of 8 responses to "Solve 2x + 3 = 7":

Response 0: "x = 2"  reward=1  advantage = 1 - 0.75 = +0.25  ✓ better
Response 1: "x = 5"  reward=0  advantage = 0 - 0.75 = -0.75  ✗ worse
Response 2: "x = 2"  reward=1  advantage = 1 - 0.75 = +0.25  ✓ better
Response 3: "x = 2"  reward=1  advantage = 1 - 0.75 = +0.25  ✓ better
Response 4: "x = 2"  reward=1  advantage = 1 - 0.75 = +0.25  ✓ better
Response 5: "x = 3"  reward=0  advantage = 0 - 0.75 = -0.75  ✗ worse
Response 6: "x = 2"  reward=1  advantage = 1 - 0.75 = +0.25  ✓ better
Response 7: "x = 2"  reward=1  advantage = 1 - 0.75 = +0.25  ✓ better

Sum of advantages ≈ 0 (this is why pg_loss ≈ 0 in logs!)
```

---

## Part 5: Loss Computation

### 5.1 The Loss Equation

```
Total Loss = pg_loss - entropy_coef × entropy + kl_coef × KL

Where:
  pg_loss      = Policy gradient loss (main RL signal)
  entropy      = How uncertain the model is (exploration)
  KL           = How different from reference model (stability)
```

### 5.2 Policy Gradient Loss

**Intuition:**

```
For each token in response:

  ratio = P_new(token) / P_old(token)    # How much did probability change?
  
  pg_loss = -clip(ratio, 0.8, 1.28) × advantage


EXAMPLE - Token "x" in correct response (advantage = +0.25):

  If ratio = 1.1 (model increased probability by 10%):
    pg_loss = -1.1 × 0.25 = -0.275  ← NEGATIVE loss
    
    Negative loss = GOOD = gradient will INCREASE this probability more!


EXAMPLE - Token "5" in wrong response (advantage = -0.75):

  If ratio = 1.0 (probability unchanged):
    pg_loss = -1.0 × (-0.75) = +0.75  ← POSITIVE loss
    
    Positive loss = BAD = gradient will DECREASE this probability!
```

### 5.3 Code: Policy Loss

**File:** `miles/utils/ppo_utils.py`

```python
@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,       # log(π_new) - log(π_old)
    advantages: torch.Tensor,    # Normalized rewards
    eps_clip: float,             # 0.2 - lower clip bound
    eps_clip_high: float,        # 0.28 - upper clip bound
):
    """
    Compute clipped policy gradient loss.
    
    Clipping prevents the policy from changing too much in one step.
    """
    # Convert log ratio to ratio
    ratio = (-ppo_kl).exp()
    
    # Unclipped loss
    pg_losses1 = -ratio * advantages
    
    # Clipped loss: ratio constrained to [0.8, 1.28]
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    
    # Take the more conservative (larger) loss
    pg_loss = torch.maximum(pg_losses1, pg_losses2)
    
    # Track how often clipping happens
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()
    
    return pg_loss, clipfrac
```

### 5.4 Code: KL Divergence

**File:** `miles/utils/ppo_utils.py`

```python
def compute_approx_kl(
    log_probs: torch.Tensor,       # log P_new
    log_probs_base: torch.Tensor,  # log P_ref (reference model)
    kl_loss_type: str,             # "low_var_kl" in your config
):
    """
    Compute KL divergence between current policy and reference.
    """
    log_ratio = log_probs.float() - log_probs_base.float()
    
    if kl_loss_type == "low_var_kl":
        # Low variance, non-negative, unbiased estimator
        # From: http://joschu.net/blog/kl-approx.html
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
        
        # Clamp for numerical stability
        kl = torch.clamp(kl, min=-10, max=10)
    
    return kl
```

### 5.5 Code: Advantage Computation

**File:** `miles/backends/megatron_utils/loss.py`

```python
def compute_advantages_and_returns(args, rollout_data):
    """
    Convert normalized rewards to per-token advantages.
    """
    rewards = rollout_data["rewards"]
    log_probs = rollout_data["log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    
    # Compute KL for monitoring
    kl = [
        compute_approx_kl(log_probs[i], ref_log_probs[i], 
                          kl_loss_type=args.kl_loss_type)
        for i in range(len(log_probs))
    ]
    
    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # GRPO: Each token gets the same advantage
        returns = get_grpo_returns(rewards, kl)
        advantages = [r for r in returns]
    
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
```

**File:** `miles/utils/ppo_utils.py`

```python
def get_grpo_returns(rewards, kl):
    """
    Expand sequence-level rewards to per-token rewards.
    
    Every token in a response gets the SAME reward.
    """
    returns = []
    for i in range(len(rewards)):
        # Create tensor filled with rewards[i] for each token
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

---

## Part 6: Training Entry Point

### 6.1 Actor Training

**File:** `miles/backends/megatron_utils/actor.py`

```python
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    """
    Main training entry point, called once per rollout.
    """
    # Step 1: Wake up model (if using memory offload)
    if self.args.offload_train:
        self.wake_up()  # Reload NCCL process groups
    
    # Step 2: Get and preprocess rollout data
    with timer("data_preprocess"):
        rollout_data = self._get_rollout_data(rollout_data_ref)
    
    # Step 3: Run training
    return self.train_actor(rollout_id, rollout_data)


def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
    """
    Perform actor model training.
    """
    # Create data iterator
    data_iterator, num_microbatches = get_data_iterator(
        self.args, self.model, rollout_data
    )
    
    # Compute advantages from rewards
    compute_advantages_and_returns(self.args, rollout_data)
    
    # Run forward-backward pass
    train_step(
        self.args,
        self.model,
        self.optimizer,
        data_iterator,
        num_microbatches,
    )
```

### 6.2 Process Group Handling

**File:** `miles/utils/reloadable_process_group.py`

```python
@staticmethod
def reload_process_groups():
    """
    Reload NCCL process groups after memory offload.
    """
    pid = os.getpid()
    reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
    logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
    
    old_new_group = old_new_group_dict.get(pid)
    for reloadable_group in reloadable_groups:
        if reloadable_group.group is not None:
            continue
        group = old_new_group(ranks=reloadable_group.group_info["ranks"], 
                              backend="nccl")
        reloadable_group.group = group
    
    # Synchronize all ranks after reload
    torch.cuda.synchronize()
    dist.barrier(group=get_gloo_group())
    
    logger.info(f"Process group reload complete and synchronized in pid {pid}")
```

---

## Part 7: Hyperparameters

### 7.1 Your GRPO Configuration

```bash
# From: scripts/run-qwen3-30B-A3B.sh

GRPO_ARGS=(
   --advantage-estimator grpo    # Use GRPO algorithm
   --use-kl-loss                 # Enable KL tracking
   --kl-loss-coef 0.00           # Don't penalize KL
   --kl-loss-type low_var_kl     # Use low-variance KL estimator
   --entropy-coef 0.00           # No entropy bonus
   --eps-clip 0.2                # Lower clip bound
   --eps-clip-high 0.28          # Upper clip bound
)
```

### 7.2 Parameter Effects

| Parameter | Value | Effect |
|-----------|-------|--------|
| `kl-loss-coef` | **0.00** | Policy can drift freely → faster learning, risk of instability |
| `entropy-coef` | **0.00** | No exploration bonus → model can become confident |
| `eps-clip` | **0.2** | Token probability can decrease by at most 20% per step |
| `eps-clip-high` | **0.28** | Token probability can increase by at most 28% per step |

### 7.3 Loss Equation with Your Settings

```
Your settings: kl_coef = 0, entropy_coef = 0

Total Loss = pg_loss - 0 × entropy + 0 × KL
           = pg_loss only

This means:
- Model learns purely from policy gradient
- Clipping (eps-clip) is the ONLY stability mechanism
- Model can drift far from reference (unbounded KL)
```

---

## Part 8: Why Things Work

### 8.1 Why pg_loss ≈ 0 (But Model Still Learns)

```
GRPO normalizes advantages to zero-mean:

8 responses: advantages = [+0.25, -0.75, +0.25, +0.25, +0.25, -0.75, +0.25, +0.25]
Sum ≈ 0

When you average pg_loss across all tokens:
  mean(pg_loss) ≈ 0  ← This is what you see in logs

BUT individual tokens still get gradients:
  Correct response tokens: negative pg_loss → probability INCREASES
  Wrong response tokens: positive pg_loss → probability DECREASES

PROOF it works:
  grad_norm = 0.05-0.2    ← Gradients ARE flowing
  eval/aime = 0.66 → 0.80 ← Model IS improving
```

### 8.2 The Credit Assignment Problem

```
ISSUE: Reward is computed from ANSWER only, but ALL tokens are trained

Response: "<think> reasoning </think> \boxed{answer}"
                    ↓                       ↓
             Gets same advantage    Reward computed here

CONSEQUENCE: If reasoning is bad but answer is lucky-correct,
             bad reasoning gets reinforced
```

**Why It Still Works:**

For complex math problems:
- P(correct | good reasoning) ≈ 90%
- P(correct | bad reasoning) ≈ 5%

Over thousands of samples, good patterns are statistically reinforced, bad patterns are statistically discouraged.

---

## Part 9: Advanced Credit Assignment

### 9.1 The Problem with Outcome Reward Models (ORM)

The current `miles` codebase uses **Outcome Reward Models (ORM)** - rewards based only on the final answer:

```
Problem: "Solve x² - 5x + 6 = 0"

Response A (Good reasoning, correct answer):
  <think>
  Factor: (x-2)(x-3) = 0
  So x = 2 or x = 3
  </think>
  \boxed{2, 3}
  → Reward = 1

Response B (Wrong reasoning, lucky answer):
  <think>
  Let me guess... maybe 2 and 3?
  </think>
  \boxed{2, 3}
  → Reward = 1  ← Same reward despite poor reasoning!

Response C (Good reasoning, calculation error):
  <think>
  Factor: (x-2)(x-3) = 0
  So x = 2 or x = 4  ← typo
  </think>
  \boxed{2, 4}
  → Reward = 0  ← Penalized despite good reasoning!
```

### 9.2 Solution: Process Reward Models (PRM)

**Process Reward Models** assign rewards to each step of reasoning, not just the final answer.

```
Response: "Solve x² - 5x + 6 = 0"

Step 1: "I need to factor the quadratic"
  → PRM Score: 0.9 (good start)

Step 2: "Looking for two numbers that multiply to 6 and add to -5"
  → PRM Score: 0.95 (correct approach)

Step 3: "Those numbers are -2 and -3"
  → PRM Score: 0.98 (correct!)

Step 4: "So (x-2)(x-3) = 0, meaning x = 2 or x = 3"
  → PRM Score: 0.99 (correct conclusion)

Final: \boxed{2, 3}
  → ORM Score: 1.0
```

### 9.3 How PRM Works

#### Training Data Generation

PRMs are trained on step-level labels. There are two approaches:

**1. Human Annotation (expensive but accurate):**
```python
# Each step is manually labeled
training_data = [
    {"step": "Factor x² - 5x + 6", "label": "correct"},
    {"step": "= (x-2)(x-3)", "label": "correct"},
    {"step": "x = 2 or x = 3", "label": "correct"},
]
```

**2. Monte Carlo Estimation (scalable):**
```python
def estimate_step_reward(problem, partial_solution, num_completions=100):
    """
    Estimate step quality by sampling completions.
    
    Intuition: A good step leads to more correct final answers.
    """
    correct_count = 0
    
    for _ in range(num_completions):
        # Complete the solution from this step
        completion = model.generate(problem + partial_solution)
        
        # Check if final answer is correct
        if is_correct(completion, ground_truth):
            correct_count += 1
    
    # Step reward = probability of reaching correct answer from here
    return correct_count / num_completions
```

#### PRM Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROCESS REWARD MODEL                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: [Problem] + [Step 1] + [Step 2] + ... + [Step n]                  │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    Transformer Backbone                          │     │
│   │                  (same as LLM, e.g., Llama)                      │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │              Step-Level Classification Head                      │     │
│   │                                                                  │     │
│   │   For each step separator token (e.g., "\n"):                   │     │
│   │     hidden_state → Linear → Sigmoid → P(correct)                │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│   Output: [0.9, 0.95, 0.98, 0.99]  (score for each step)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Using PRM in Training

#### Reward Aggregation Strategies

```python
def aggregate_prm_rewards(step_rewards, strategy="min"):
    """
    Aggregate step-level rewards into sequence-level reward.
    """
    if strategy == "min":
        # Weakest link: sequence quality = worst step quality
        return min(step_rewards)
    
    elif strategy == "product":
        # Probability chain: P(all correct) = ∏ P(step_i correct)
        return math.prod(step_rewards)
    
    elif strategy == "last":
        # Only use final step (similar to ORM but for steps)
        return step_rewards[-1]
    
    elif strategy == "weighted_sum":
        # Later steps matter more
        weights = [0.1, 0.2, 0.3, 0.4]  # example
        return sum(w * r for w, r in zip(weights, step_rewards))
```

#### Integration with GRPO

```python
# Current miles approach (ORM):
reward = get_deepscaler_rule_based_reward(response, label)  # 0 or 1

# With PRM:
step_rewards = prm.score_steps(problem, response)  # [0.9, 0.95, 0.98, 0.99]
reward = aggregate_prm_rewards(step_rewards, strategy="min")  # 0.9
```

### 9.5 Best-of-N with PRM (Inference Time)

PRMs are especially powerful at inference time for selecting the best solution:

```python
def best_of_n_with_prm(problem, model, prm, n=64):
    """
    Generate N solutions and select the best one using PRM.
    """
    candidates = []
    
    for _ in range(n):
        response = model.generate(problem)
        step_rewards = prm.score_steps(problem, response)
        
        # Use min reward as quality indicator
        quality = min(step_rewards)
        candidates.append((response, quality))
    
    # Return the response with highest quality
    return max(candidates, key=lambda x: x[1])[0]
```

### 9.6 Comparison: ORM vs PRM

| Aspect | ORM (Current) | PRM (Advanced) |
|--------|---------------|----------------|
| **Granularity** | Sequence-level | Step-level |
| **Training Data** | Easy (just final correctness) | Hard (need step labels) |
| **Credit Assignment** | Poor (all tokens same reward) | Good (each step scored) |
| **Robustness** | Lucky guesses rewarded | Lucky guesses penalized |
| **Computation** | Fast | Slower (score each step) |
| **Best Use** | Fast iteration | High-stakes, quality focus |

### 9.7 Key Papers and Resources

#### 1. "Let's Verify Step by Step" (OpenAI, 2023)
- **Link:** https://arxiv.org/abs/2305.20050
- **Key Contribution:** First large-scale study of PRMs for math
- **Finding:** PRM + Best-of-N achieves 78% on MATH vs 72% for ORM

```
Key Results from Paper:
┌────────────────────────────────────────────────────────┐
│ Method                        │ MATH Accuracy          │
├────────────────────────────────────────────────────────┤
│ Greedy Decoding               │ 62%                    │
│ Best-of-N (ORM)               │ 72%                    │
│ Best-of-N (PRM) ← Best        │ 78%                    │
└────────────────────────────────────────────────────────┘
```

#### 2. "Math-Shepherd: Verify and Reinforce LLMs Step-by-step" (2023)
- **Link:** https://arxiv.org/abs/2312.08935
- **Key Contribution:** Automatic step-level label generation using Monte Carlo

```python
# Math-Shepherd's automatic labeling approach:
def label_step_monte_carlo(problem, steps_so_far, remaining_completions=10):
    """
    Label a step by completing the solution multiple times.
    """
    success_count = 0
    
    for _ in range(remaining_completions):
        # Complete from current point
        completion = model.complete(problem + steps_so_far)
        final_answer = extract_answer(completion)
        
        if is_correct(final_answer, ground_truth):
            success_count += 1
    
    # Step is "good" if success rate > threshold
    success_rate = success_count / remaining_completions
    return "correct" if success_rate > 0.5 else "incorrect"
```

### 9.8 Future Direction: Token-Level Rewards

Even more fine-grained than step-level:

```
"Factor x² - 5x + 6 = (x - 2)(x - 3)"
        ↑    ↑   ↑     ↑   ↑   ↑   ↑
        │    │   │     │   │   │   └── reward: 0.99
        │    │   │     │   │   └────── reward: 0.98
        │    │   │     │   └────────── reward: 0.95
        │    │   │     └────────────── reward: 0.90
        │    │   └──────────────────── reward: 0.85
        │    └──────────────────────── reward: 0.80
        └───────────────────────────── reward: 0.70
```

This is an active research area with approaches like:
- **RLHF with dense rewards**
- **Process supervision at token level**
- **Reward modeling with attention attribution**

---

## Part 10: Code Reference

### Quick Navigation

| What | File | Key Function |
|------|------|--------------|
| **Reward computation** | `miles/rollout/rm_hub/deepscaler.py` | `get_deepscaler_rule_based_reward()` |
| **Answer grading** | `miles/rollout/rm_hub/math_utils.py` | `grade_answer_mathd()`, `grade_answer_sympy()` |
| **Reward routing** | `miles/rollout/rm_hub/__init__.py` | `rule_based_rm()` |
| **GRPO normalization** | `miles/ray/rollout.py` | `_reward_post_process()` |
| **Advantage computation** | `miles/backends/megatron_utils/loss.py` | `compute_advantages_and_returns()` |
| **Policy loss** | `miles/utils/ppo_utils.py` | `compute_policy_loss()` |
| **KL computation** | `miles/utils/ppo_utils.py` | `compute_approx_kl()` |
| **Training entry** | `miles/backends/megatron_utils/actor.py` | `train()`, `train_actor()` |
| **Process groups** | `miles/utils/reloadable_process_group.py` | `reload_process_groups()` |

### Code Flow Diagram

```
train.py
    │
    ├──→ ray/rollout.py ──→ Generate responses
    │         │
    │         ├──→ rm_hub/deepscaler.py ──→ Compute rewards (0 or 1)
    │         │
    │         └──→ _reward_post_process() ──→ GRPO normalize → advantages
    │
    └──→ megatron_utils/actor.py:train()
              │
              ├──→ wake_up() ──→ reloadable_process_group.py
              │
              ├──→ loss.py:compute_advantages_and_returns()
              │         │
              │         └──→ ppo_utils.py:get_grpo_returns()
              │
              └──→ ppo_utils.py:compute_policy_loss()
                        │
                        └──→ Backprop → optimizer.step()
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GRPO TRAINING CHEAT SHEET                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LOSS = pg_loss - entropy_coef × H + kl_coef × KL                          │
│         ───────   ─────────────────   ──────────────                        │
│         minimize  maximize            minimize                              │
│         (= max    (exploration)       (stay close                          │
│          reward)                       to reference)                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  REWARD FLOW:                                                               │
│                                                                             │
│  Response → Extract \boxed{} → Grade → 0 or 1                              │
│          → Normalize in group → Advantage (+/-)                            │
│          → Apply to ALL tokens → pg_loss                                   │
│          → Backprop → Update model                                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  KEY FILES:                                                                 │
│                                                                             │
│  rm_hub/deepscaler.py     → Reward function                                │
│  rm_hub/math_utils.py     → Answer grading                                 │
│  ray/rollout.py           → GRPO normalization                             │
│  megatron_utils/loss.py   → Advantage computation                          │
│  utils/ppo_utils.py       → Policy loss & KL                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  CREDIT ASSIGNMENT:                                                         │
│                                                                             │
│  ORM (current):  Final answer only → same reward for all tokens           │
│  PRM (advanced): Each step scored → better credit assignment              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  DEBUGGING TIPS:                                                            │
│                                                                             │
│  • pg_loss ≈ 0 is NORMAL (advantages sum to 0)                             │
│  • Check grad_norm > 0 to verify learning                                  │
│  • Check eval metrics to verify improvement                                │
│  • kl_loss increasing = policy drifting (expected with kl_coef=0)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Advantage** | How much better/worse a response is compared to average |
| **Clipping** | Constraining the policy ratio to prevent large updates |
| **Entropy** | Measure of model uncertainty/randomness |
| **GRPO** | Group Relative Policy Optimization |
| **KL Divergence** | Measure of how different current policy is from reference |
| **ORM** | Outcome Reward Model - rewards based on final answer only |
| **PRM** | Process Reward Model - rewards based on each reasoning step |
| **Policy Gradient** | Method to optimize policy by gradient ascent on expected reward |
| **Ratio** | P_new(token) / P_old(token) - how much probability changed |
| **Rollout** | Generating responses from the current policy |

---

*Document generated from miles codebase analysis. Last updated: January 2026.*


