LLMROUTER: UNIFIED INFRASTRUCTURE
FOR DEVELOPING, EVALUATING, AND DEPLOYING
LLM ROUTERS
Tao Feng1*, Fangxu Yu2*, Haozhen Zhang3*, Zhongjie Dai1
, Liangqi Yuan4
, Zijie Lei1
,
Weizhi Zhang5
, Kunlun Zhu1
, Haodong Yue1
, Keyang Xuan1
, Ge Liu1
, Jiaxuan You1
1University of Illinois Urbana-Champaign, 2University of Maryland, College Park,
3Nanyang Technological University, 4Purdue University, 5University of Illinois Chicago
 Project xRouteBench Code
ABSTRACT
No single large language model (LLM) is optimal across all queries and budget
constraints, making model routing essential for cost-effective LLM deployment.
Existing routers span binary quality predictors, cost-aware cascades, graph-based
routers, and agentic routers, yet their diverse formalisms and incompatible implementations, coupled with the absence of a standardized evaluation pipeline,
hinder fair comparison and further extension. In this paper, we present a unified formulation of LLM routing as a sequential decision process. Under this
formulation, a router can be characterized in terms of five types of components:
context encoders, model encoders, scoring functions, decision rules, and learning
signals. Existing methods can then be organized into three families of single-turn,
multi-turn, and personalized routing. Building on this formulation, we develop an
automated pipeline that constructs routing supervision by systematically running
a pool of candidate models across benchmarks and evaluates routers in terms of
both response quality and inference cost under a unified protocol. The resulting
benchmark, xRouteBench, spans generic LLM tasks, memory-augmented, vision
(image and video), time-series, and personalized routing scenarios. Grounded in
the formulation and pipeline, we present LLMRouter, an open-source infrastructure
for standardized and modular implementation of LLM routers, where users can
add a new router by implementing only a routing method and a loss function and
access built-in implementations of more than 16 representative routers spanning all
three families. Using the library and benchmark, we conduct a systematic empirical study of LLM routing and find that learned routers achieve a 14.6% relative
improvement over the strongest fixed-model baseline, router rankings reverse in
favor of lightweight designs under tighter cost constraints, and user-conditioned
routing delivers consistent personalization gains.
1 INTRODUCTION
The rapid proliferation of large language models (LLMs) has created a heterogeneous ecosystem of
models with widely varying costs and task-specific capabilities, ranging from frontier systems to
substantially cheaper open-weight alternatives. Since no single model is optimal across all queries
and budget constraints, model routing, which determines which model should handle each query,
has become essential for cost-effective LLM deployment. Beyond cost efficiency, routing also
matches each query to the candidate model best suited to it and adapts model choice to user-specific
preferences (Figure 1). Rich research has been devoted to this problem, from binary routers that
arbitrate between a weak and a strong model (Ding et al., 2024; Ong et al., 2024) and cost-aware
cascades (Chen et al., 2023; Aggarwal et al., 2024), to reward-guided ensembles, contrastive and
graph-based routers (Chen et al., 2024; Feng et al., 2024), personalized routers that adapt to individual
*Equal contribution.
arXiv:2608.06867v1 [cs.CL] 7 Aug 2026
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
users (Xie et al., 2025; Dai et al., 2025), and agentic routers trained with reinforcement learning
(Zhang et al., 2025; Feng et al., 2026).
Despite this rapid progress, the field still lacks a unified foundation on which these diverse approaches
can be developed and compared, mainly due to two obstacles. First, existing routers are developed
under distinct formalisms, released as separate codebases with incompatible interfaces, trained with
different supervision, and tuned for different candidate pools, making it difficult to isolate the design
elements that actually drive performance or determine whether observed differences stem from
the routers themselves or from their broader experimental stacks. Second, evaluating a router is
fundamentally more demanding than evaluating a single model, as constructing routing supervision
and enabling standardized evaluation require running every candidate model on every benchmark
query and scoring each response with task-specific metrics. Existing benchmarks precompute
candidate responses for fixed model pools (Hu et al., 2024; Huang et al., 2025b), but are limited
to single-turn routing and provide no pipeline for generating supervision for new benchmarks or
candidate pools. Consequently, multi-turn and personalized routing still lack a standardized, costaware evaluation framework, making routing methods difficult to compare, reuse, and transfer from
offline studies to real applications.
In this paper, we introduce LLMRouter, a unified infrastructure for developing, evaluating, and deploying routing policies over heterogeneous LLM backends. Within this infrastructure, a router is
characterized by five types of components: context encoders, model encoders, scoring functions,
decision rules, and learning signals. This abstraction accommodates existing routers, which we
group into three families of single-turn, multi-turn (including agentic), and personalized routing. It
also reduces the effort to add a new router to implementing a routing method and a loss function,
while data construction, training, inference, and evaluation apply unchanged, so switching the router,
candidate pool, or training objective requires only a configuration change rather than reimplementation. LLMRouter includes more than 16 representative routers spanning all three families, and can
expose any of them as an OpenAI-compatible server for deployment on messaging platforms via
OpenClaw (OpenClaw, 2026) or through a ComfyUI-based visual interface for code-free prototyping.
LLMRouter further automates the construction of routing supervision and evaluation, the main obstacle
to comparing routers on equal footing. Its pipeline assembles queries from established benchmarks,
dispatches each query to a pool of 18 candidate models spanning a broad price range, and scores every
response with task-specific metrics while recording token-level cost. Every router is then evaluated
on the same queries, candidate pool, and metrics, enabling direct comparison of their quality-cost
trade-offs. With this pipeline, we construct xRouteBench, a benchmark that spans generic LLM
tasks, memory-augmented, vision (image and video), time-series, and personalized scenarios under
one protocol.
Leveraging LLMRouter and xRouteBench, we conduct a systematic empirical study of LLM routing
across the three families under one protocol. Our study surfaces four findings: (i) no single router
dominates, as the best router varies across tasks and cost budgets. Strong average performance
therefore reflects consistency across scenarios rather than dominance in any single setting. (ii)
learned routing still outperforms the strongest fixed-model baseline, because always selecting
the largest model incurs the highest cost yet delivers only mediocre performance, whereas learned
routers select smaller, cheaper models for many queries that the largest model answers incorrectly.
(iii) multi-turn routing does not consistently outperform single-turn routing, as additional rounds
of decomposition and aggregation often add cost and redundant information, and their benefit hinges
on the capability of the base model that performs them. (iv) personalization pays off, but only when
user context is modeled well, as a user-conditioned router ranks first under both the persona judge
and real human preferences, yet the two settings favor different personalized designs. We also release
the library and benchmark in the hope of fostering more systematic progress in LLM routing.
2 A UNIFIED FORMULATION OF LLM ROUTING
2.1 ROUTING AS A SEQUENTIAL DECISION PROCESS
Existing routers take seemingly incompatible forms, from binary routers that arbitrate between a
weak and a strong model (Ding et al., 2024; Ong et al., 2024) and cost-aware cascades (Chen et al.,
2023; Aggarwal et al., 2024) to graph-based routers (Feng et al., 2024; Yu et al., 2026a) and agentic
2
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
... LLM Candidates
Context
Encoder
Model
Encoder
Single-turn
Query Persona History
Multi-turn
Query Persona History
Personalized
Query Persona History
Routing State
Selected
LLM
Response
#N
Selected
LLM #N
Answer
......
Answer
Single-turn
Multi-turn
...... ......
Routing
State for
Routers
Router Response
#1
Selected
LLM #1
Query
Sub-query
#N
History
Sub-query
#1
Easy Problems
Hard Problems
Small LLM
Math LLM
Route
Large LLM
Route
Cost- Efficient
High
Performance
&
Code LLM
1. Routing for cost efficiency
2. Routing for capability matching
3. Routing for user preference
Llama
User query Qwen
Persona
Prefer
Disfavor
Skilled
Unskilled
Math Problems
Unified Formulation of LLM Routing
Figure 1: Overview of LLM routing. Routing is driven by three needs (left), namely cost efficiency,
capability matching, and user preference. Our unified formulation (right) casts all of them as one
decision process: a context encoder Eq represents the routing state of query, persona, and interaction
history, a model encoder Em represents each candidate, and the router dispatches the query or its
sub-queries to selected models and aggregates their responses into the answer. The single-turn,
multi-turn, and personalized families differ only in which part of the state they observe.
routers trained with reinforcement learning (Zhang et al., 2025), yet they can all be formulated as a
sequential decision process. At step t, the router observes a state st “ pq, u, htq, consisting of the
input query q, an optional user context u (e.g., a user identifier with past interactions and feedback),
and the interaction history ht accumulated so far, and takes an action at P M Y tKu. The dispatch
action at “ m sends the state to candidate m from the pool M “ tm1, . . . , mKu and appends its
response yt to the history, ht`1 “ ht ‘ yt, while the terminating action at “ K ends the episode and
aggregates the collected responses into the final answer y; single-turn routing is the special case that
terminates after one dispatch. The goal of routing is a policy π whose trajectory τ “ pa1, . . . , aT q
produces high-quality answers at low inference cost:
π
‹ “ arg max
π
Eq, τ„π
“
perfpy | qq ´ λ ¨ cpτ q
‰
, (1)
where perfpy | qq aggregates task-specific quality metrics (e.g., accuracy, F1, or an LLM-judged
score), cpτ q sums the monetary or token cost of every call in the routing trajectory τ , and λ ě 0
controls the performance–cost trade-off. Under this formulation, a router is characterized by the
choice of a context encoder Eq that encodes the routing state, a model encoder Em that encodes
each LLM candidate, a scoring function g and a decision rule d that turn the context and model
representations into a routing action, and a learning signal L that fits these components toward the
optimal policy. This section elaborates on each component and demonstrates how existing routers
fall into three families, namely single-turn, multi-turn (including agentic), and personalized, with
specific designs of these five components (Table 1). We provide an overview in Figure 1.
Context encoder. The context encoder Eq maps the routing state st to the representation on which
the routing decision is based, and its output takes one of two forms. i) Embedding-based: the state
is represented as a vector. Rating-based routers degenerate to a constant that ignores the query and
routes by global model quality, kNN-style routers use an off-the-shelf sentence embedding (Hu et al.,
2024; Shnitzer et al., 2023), discriminative routers train a lightweight encoder over frozen embeddings
(Ding et al., 2024; Stripelis et al., 2024), and personalized routers condition the representation on
user and session nodes of a heterogeneous interaction graph (Xie et al., 2025; Dai et al., 2025).
ii) Text-based: the state is kept in natural language. Cascades append the draft response and a
verification confidence to the query (Chen et al., 2023; Aggarwal et al., 2024), and fine-tuned LM
routers, exemplified by Router-R1, verbalize the whole state directly in the prompt, leaving its
representation to the model’s forward pass (Ong et al., 2024; Zhang et al., 2025). Which portion of
the state Eq reads is precisely what separates the three router families, and any router is personalized
by swapping in a user-conditioned Eq while inheriting the remaining components.
Model encoder. The model encoder Em encodes each candidate in the pool. i) Static metadata:
the simplest choice describes a candidate by its model size, capability description, and pricing. ii)
Historical profiles: most routers instead profile candidates by their past behavior, representing a
model by the set of embedded queries it has previously solved (kNN), a scalar rating (Elo), or a
latent factor fit by matrix factorization. iii) Learned embeddings: stronger routers learn model
3
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Table 1: Instantiation of the unified routing formulation for the three router families. For
each family, the table specifies the routing state s, the context and model encoders Eq and Em, the
routing action defined by the scoring function g and decision rule d, and the learning signal L used to
optimize response quality and inference cost.
Family State s Encoders Eq, Em Routing action (scoring g, decision d) Learning signal L (surrogate of Eq. 1)
Single-turn pqq Eqpqq, Empmq a “ arg maxmPM g
`
Eqpqq, Empmq
˘
fit g to per-candidate reward perfpym | qq ´ λ cm
Multi-turn pq, htq Eqpq, htq, Empmq at „ d
`
tgpEqpq, htq, Empmqqum
˘
maximize episode return Eτ
“
perfpy | qq ´ λ cpτ q
‰
Personalized pq, u, htq Eqpq, u, htq, Empmq a “ arg maxmPM g
`
Eqpq, u, htq, Empmq
˘
fit g to comparisons m` ąu m´ observing perfu
embeddings jointly with the context encoder (Chen et al., 2024; Feng et al., 2024; Zhuang et al.,
2025). iv) Verbalized description: fine-tuned LM routers instead name the candidates directly in the
prompt (Ong et al., 2024; Zhang et al., 2025).
Scoring function and decision rule. The scoring function g measures the compatibility between the
encoded state and each candidate, and the decision rule d converts the resulting scores into a routing
action. Instantiations of g track the encoders, from embedding similarity in kNN-style routers and
a bilinear product in factorization-based ones, to a classification head (Ding et al., 2024; Stripelis
et al., 2024), message passing over a query–model graph (Feng et al., 2024), and next-token logits in
fine-tuned LM routers that fold Eq, Em, and g into one forward pass (Ong et al., 2024). For d, greedy
arg max is the default choice, yet it is optimal for Eq. 1 only when λ “ 0. Cost-aware rules instead
threshold the predicted quality gap between a cheap and an expensive model (Ding et al., 2024),
accept or escalate in cascades (Chen et al., 2023; Aggarwal et al., 2024), or sample for exploration in
online settings (Dai et al., 2024), and multi-turn routers further equip d with the terminating action K
(Zhang et al., 2025).
Learning signal. The learning signal L specifies how the components above are fit toward Eq. 1.
Non-parametric routers require no training and rely purely on stored interactions. Supervised routers
fit pointwise correctness labels harvested by running the candidate pool over benchmark queries,
preference-based routers learn from pairwise comparisons such as human votes from Chatbot Arena
(Ong et al., 2024) or contrastive objectives that pull queries toward the models that solve them
(Chen et al., 2024), and agentic routers directly optimize trajectory-level rewards with reinforcement
learning (Zhang et al., 2025). In every case, L is a surrogate for the same objective. What differs is not
the goal but the form in which perf is observable, measured for every candidate by supervised routers,
returned only at the end of a trajectory for agentic ones, and revealed only through comparisons when
quality is user-specific.
2.2 AUTOMATIC EVALUATION OF LLM ROUTING
Evaluating a router is substantially more demanding than evaluating a single model. Under Eq. 1,
a router must be judged on both the quality of its answers and the cost spent to obtain them,
and constructing its supervision requires knowing how every candidate performs on every query
under task-specific metrics. In current practice, these elements are assembled manually for a single
benchmark and fixed candidate pool (Hu et al., 2024; Huang et al., 2025b), requiring fresh engineering
for every new task or candidate pool.
LLMRouter automates this process end-to-end with a three-stage pipeline: i) Query Curation: queries
are sampled from source benchmarks, normalized into a unified schema, and split into training and
test sets; ii) Response Collection: each query is dispatched to every candidate in the pool, which is
declared in a single configuration file, and responses are collected together with their token counts;
iii) Metric Scoring and Pricing: every response is scored with its task metric and priced from its
token counts. The product is a dense query–model matrix of performance and cost that serves at
once as routing supervision and as the test bed, so a new task or candidate pool enters through a
configuration change rather than a re-engineered stack.
Evaluation reuses this path, except that a test query goes only to the candidate the router selects rather
than to the whole pool. Every router therefore faces the same queries, candidate pool, and metrics, so
measured differences reflect the routing policy rather than the surrounding stack. For the multi-turn
and agentic families, every decomposition and aggregation call is priced into the trajectory cost.
4
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Metrics. Task quality is assessed using built-in metrics aligned with standard benchmark conventions,
including exact and close matching, multiple-choice accuracy, token-level F1, mathematical answer
verification, and execution-based code evaluation. LLMRouter also supports optional LLM-based
judging and exposes a weighted objective that balances performance and cost, allowing routers and
trainers to target performance-first, cost-sensitive, or hybrid operating points.
3 XROUTEBENCH: A MULTI-SCENARIO BENCHMARK FOR LLM ROUTING
Existing routing benchmarks cover only a subset of the settings captured by the formulation in §2.1.
RouterBench (Hu et al., 2024) precomputes candidate responses for single-turn text queries over a
fixed candidate pool, but does not cover settings in which input-token costs dominate or only a subset
of candidate models can process the input. RouterEval (Huang et al., 2025b) aggregates large-scale
performance records but likewise focuses on single-turn text tasks and evaluates response quality
independently of inference cost. Recent vision–language routing benchmarks (Huang et al., 2025a;
Ma et al., 2026) extend routing evaluation to image inputs, but remain limited to image question
answering and do not cover video, long-context, or modality-selection settings. Preference data from
Chatbot Arena (Zheng et al., 2023) provides population-level signals but lacks the persistent user
context needed to supervise user-conditioned routing. To close these gaps, we construct xRouteBench
to evaluate, under a unified cost-aware protocol, regimes in which routing decisions fundamentally
differ: long-context inputs for which input-token costs dominate, image and video inputs that only a
subset of candidate models can process, time-series inputs with multiple modality encodings, and
tasks with user-specific quality preferences.
Math
GSM8K
10.5%
MATH
10.5%
AIME
0.3%
Code
MBPP
10.5%
HumanEval 0.3%
Common
sense
ARC-C
10.5%
OpenBookQA
10.5%
Commonsense QA
1.0%
HellaSwag
1.0%
Reading SQuAD
1.0%
BoolQ
1.0%
Knowledge
MMLU
10.5%
10.5%
MMLU-Pro
Memory
LoCoMo
6.6%
LongMemEval 2.1%
Personalized
Chatbot Arena
6.3%
MT-Bench
0.2%
TimeSeries
TSRBench
2.7%
Multimodal
MathVista
2.1%
Geometry3K
1.3%
Charades-Ego
0.6%
xRouteBench
Figure 2: Task composition of xRouteBench.
The benchmark covers generic LLM tasks, memory, vision, time-series, and personalized routing,
with percentages indicating the proportion of test
queries contributed by each dataset.
Design principle. All tasks are constructed by
the LLMRouter data engine and share a common
query schema, supervision format, and evaluation protocol. Each non-text asset is converted
by a transformation script into a self-contained
textual query, with an optional pointer to the
source image, video, or time series. This separates routing from perception by ensuring that
text-only and multimodal candidates receive the
same textual input. For every query, the protocol jointly evaluates response quality and inference cost, exposing regimes in which inputtoken costs dominate answer-generation costs.
Adding a new application requires only a transformation script and a registered metric.
Tracks. xRouteBench spans five tracks comprising 4,767 instances. Figure 2 provides
an overview of task distribution. Specifically,
we include: (i) Generic LLM Tasks mix established knowledge and commonsense QA
(MMLU (Hendrycks et al., 2020), MMLU-Pro
(Wang et al., 2024), ARC-Challenge (Clark
et al., 2018), OpenBookQA (Mihaylov et al.,
2018), CommonsenseQA (Talmor et al., 2019),
BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), SQuAD (Rajpurkar et al., 2016)),
mathematical reasoning (GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), AIME), and
code generation (MBPP (Austin et al., 2021), HumanEval (Chen, 2021)), the conventional single-turn
text setting. (ii) Memory routes long-horizon conversational QA over hundreds of accumulated turns
(LoCoMo (Maharana et al., 2024), LongMemEval (Wu et al., 2024)), where token cost is governed
by the history rather than the answer. (iii) Vision covers image-grounded mathematical reasoning
(Geometry3K (Lu et al., 2021), MathVista (Lu et al., 2024b)) and egocentric video understanding
(Charades-Ego (Sigurdsson et al., 2018)). (iv) TimeSeries covers time-series reasoning (TSRBench
(Yu et al., 2026b)), with each series rendered as both text and image so the router also selects a modality encoding. (v) Personalized draws open-ended prompts from Chatbot Arena and MT-Bench (Zheng
5
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Route Engine
Router Trainer
Objective
Definition
Training
Execution
Encoded
States
Supervision
Type
Trained
Router
Candidate
Scoring
Routing
Decision
Dispatch /
Aggregate
Deployment
CommandLine Interface
OpenClaw
Server
ComfyUI
Interface
Multi-Agent
System
Evaluation
xRouteBench Router
Evaluation
Router Library
Router
Pool
LLM
Pool
Data Engine
Query
Curation
Response
Collection
Scoring &
Pricing
Figure 3: Architecture of LLMRouter. The system consists of six modules that support routing data
construction, router implementation and training, inference, evaluation, and deployment.
et al., 2023), each tied to a user persona and scored by a persona-conditioned LLM judge, so its
supervision is preference feedback rather than pointwise correctness.
4 THE LLMRouter LIBRARY
LLMRouter ties the formulation of §2.1, the evaluation protocol of §2.2, and xRouteBench into one
executable system organized as six modules around a single query–model matrix (Figure 3). Its
organizing principle is that the five components of a router are the only thing a user writes, while data
construction, training, inference, evaluation, and deployment are shared infrastructure that operates
on any router unchanged. Swapping a router, a candidate pool, or a training objective is therefore a
configuration change rather than a reimplementation.
Data Engine. The data engine implements the three-stage construction pipeline of §2.2, turning a
declared task list and candidate pool into the query–model matrix that supervises and tests every
router. Adding a task requires only a prompt template and a registered metric, and adding a candidate
requires only its endpoint and per-token price.
Router Library. The library implements more than 16 routers spanning all three families under
a unified MetaRouter interface, which hides mechanisms as different as nearest-neighbor retrieval
in a kNN router and autoregressive decoding in a fine-tuned LM router behind one call. Adding a
new router requires subclassing MetaRouter and implementing either route_single or its batched
counterpart, route_batch. Within this method, the context encoder Eq, model encoder Em, scoring
function g, and decision rule d map a routing state to a routing action. A short YAML file specifies
the router’s candidate pool and objective weights, after which the router can be invoked by name
using the same commands as any built-in method. Personalization and component ablations can then
be performed with a one-line change, without forking the codebase. Figure 4 shows the complete
code needed to design a new router.
Trainer. Training is decoupled from routing through a BaseTrainer. Its loss_func defines the
learning signal L as a pointwise loss, pairwise loss, or trajectory-level reward, while its train loop
uses this signal to optimize the router for the weighted objective in Eq. 1. A router and its trainer are
paired but can be swapped independently, allowing the same scorer to be trained with different forms
of supervision without modifying its routing code. Non-parametric routers bypass this module.
Route Engine. At inference time, the route engine drives any router through the same call, dispatching
the query to the selected candidate. For multi-turn policies, it repeats the decision step until a
termination action is produced and aggregates the collected responses into the final answer.
Evaluation. The evaluation module implements the evaluation protocol of §2.2, scoring each router
on the same test queries, candidate pool, and metrics, and sweeping the trade-off weight λ to trace its
performance–cost frontier.
6
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
from llmrouter.models import MetaRouter, BaseTrainer
class MyRouter(MetaRouter): # (E_q, E_m, g, d): state -> action
def route_single(self, query):
s = self.encode_state(query) # context encoder E_q
scores = self.score(s, self.models) # model encoder E_m + scoring g
query["model_name"] = self.decide(scores) # decision rule d
return query
class MyRouterTrainer(BaseTrainer): # learning signal L
def loss_func(self, outputs, batch):
return my_objective(outputs, batch) # pointwise / pairwise / RL reward
# Train and run through the same interface as every built-in router.
router = MyRouter(yaml_path="my_router.yaml")
trainer = MyRouterTrainer(router)
trainer.train()
answer = router.route_single({"query": "..."})
Figure 4: The five components of the routing formulation map onto two classes in LLMRouter. A
router subclasses MetaRouter and implements route_single (or route_batch), where the context
encoder Eq, model encoder Em, scoring function g, and decision rule d turn a state into a selected
model; the learning signal L lives in a BaseTrainer subclass.
Deployment. In addition to training and inference commands, LLMRouter can expose any router as
an OpenAI-compatible server that integrates with OpenClaw (OpenClaw, 2026) for deployment on
messaging platforms such as Slack and Discord. A routing memory persists the interaction history
h across turns, while a ComfyUI-based canvas supports code-free prototyping. The same router
evaluated offline can therefore serve live single-agent and multi-agent traffic without modification.
5 EXPERIMENTS
5.1 EXPERIMENTAL SETUPS
Benchmarks. We evaluate routers across the five xRouteBench tracks defined in §3: Generic LLM
Tasks, memory, vision, time-series, and personalized. Together, they comprise eight test sets, with
full statistics reported in Table 6. For each query in the memory track, we retrieve up to five memory
items as context and score responses using token-level F1.
LLM Candidates. The candidate pool contains 18 open-weight models served through two providers
(i.e., Together API1
and NVIDIA NIM API2
), spanning 7B to 671B parameters. It covers Gemma-2-
9B (Team et al., 2024); Mistral-7B, Mistral-Small-24B, Mixtral-8x7B, and Mixtral-8x22B (Jiang
et al., 2023; 2024); Qwen2.5-7B, Qwen3-Next-80B, and Qwen3-Coder (Yang et al., 2024; 2025);
Llama-3-8B, Llama-3-70B, Llama-3.3-70B, and Llama-4-Maverick (Grattafiori et al., 2024; Adcock
et al., 2026); GPT-OSS-20B and GPT-OSS-120B (Agarwal et al., 2025); RNJ-1-15B (Callahan et al.,
2026); and the two 671B models DeepSeek-V3.1 (Liu et al., 2024) and Cogito-v2 (Deep Cogito,
2025). Per-token pricing is given in Table 9 in the appendix.
Implemented Routers. LLMRouter implements more than 16 routers covering three families. (i)
Single-turn routers include kNNRouter (Li, 2025), SVMRouter, MLPRouter, EloRouter, and
MFRouter (Ong et al., 2024; Shnitzer et al., 2023), as well as RouterDC (Chen et al., 2024),
Hybrid LLM (Ding et al., 2024), AutoMix (Aggarwal et al., 2024), GraphRouter (Feng et al.,
2024), CausalLM (Ong et al., 2024), and two rule-based baselines that always select the smallest or
largest model; (ii) Multi-turn routers include Router-R1 (Zhang et al., 2025) together with kNNbased and LLM-based multi-round routers; and (iii) Personalized routers include GMTRouter (Xie
et al., 2025) and PersonalizedRouter (Dai et al., 2025).
1
https://www.together.ai/
2
https://build.nvidia.com/
7
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Table 2: Results on xRouteBench under the performance-first setting pα, βq “ p1.0, 0.0q. Scores
are reported across the Generic LLM Tasks, memory, vision, and time-series tracks, together with
their average. Following the original implementations where applicable, all multi-turn routers use
Qwen2.5-3B-Instruct as the base model. Top two results are highlighted in bold and underline.
Router Generic LLM Tasks Memory Vision TimeSeries Avg
LoCoMo LongMemEval Geometry3K MathVista Video
Rule-based baselines
Smallest-LLM 57.55 25.44 36.77 27.87 35.00 33.33 49.61 37.94
Largest-LLM 70.29 26.59 35.57 37.70 33.00 22.22 45.67 38.72
Single-turn routers
kNNRouter 71.37 25.24 38.74 31.15 41.00 29.63 51.97 41.30
SVMRouter 74.21 27.64 38.68 42.62 47.00 29.63 55.91 45.10
MLPRouter 68.12 26.78 32.27 27.87 34.00 29.63 56.69 39.34
MFRouter 67.23 24.49 34.91 40.98 29.00 22.22 51.97 38.69
EloRouter 64.15 25.70 37.27 45.90 50.00 25.93 63.78 44.68
Hybrid LLM 64.68 25.89 36.56 32.79 37.00 33.33 51.18 40.20
RouterDC 80.56 24.93 36.77 16.39 24.00 25.93 45.67 36.32
GraphRouter 80.54 25.94 33.93 42.62 50.00 22.22 62.99 45.46
CausalLM 66.90 25.40 37.60 24.60 34.00 33.33 45.70 38.22
Multi-turn routers
Router-R1 35.64 24.60 17.28 14.75 18.00 22.22 23.62 22.30
kNN-MultiRound 13.99 24.70 18.32 16.39 30.00 25.93 33.07 23.20
LLM-MultiRound 12.98 24.60 17.44 14.29 31.03 25.93 30.33 22.37
Evaluation Protocol. We score each router by a weighted reward α ¨ perf ´ β ¨ cost. We sweep
five weight settings from the quality-only pα, βq “ p1.0, 0.0q to the heavily cost-weighted p0.2, 0.8q.
Multi-round and RL-based routers cannot optimize this weighted objective and are therefore run once
under a single configuration. On the personalized track, answers are scored by a persona-conditioned
LLM judge (DeepSeek-V3.1) as win, tie, or loss (1, 0.5, 0), and the judge’s own cost is excluded
from the reported cost.
5.2 MAIN RESULTS
Table 2 and Table 3 report the results under the performance-only setting. Based on these results, we
have the following key observations:
No single router dominates across all tasks: The winner router varies across tasks. For example,
RouterDC performs best on Generic LLM Tasks and SVMRouter on LoCoMo. Though GraphRouter
attains the best average on xRouteBench, it did not consistently outperform other routers in all tasks.
Multi-turn routing does not consistently outperform single-turn routing: Across all benchmarks,
multiple rounds of routing and aggregation provide no consistent gain over a single routing decision.
For many queries, one well-chosen route is sufficient, whereas additional rounds introduce redundant
information and computational overhead. Multi-turn routers also rely on a base model (Qwen2.5-3BInstruct) to decompose queries and aggregate responses, making their performance sensitive to the
capabilities of this model. These results highlight the need for better sufficiency estimation, early
stopping, and more effective decomposition and aggregation.
Table 3: Performance comparison on the personalized track. Top two results are highlighted in bold
and underline.
Router Acc. Router Acc.
GMTRouter 68.78 RouterDC 56.44
PersonalizedRouter 67.86 MFRouter 54.39
EloRouter 66.40 MLPRouter 52.93
GraphRouter 65.23 kNNRouter 51.76
SVMRouter 65.08 CausalLM 46.78
Largest-LLM 58.05 Router-R1 45.46
Hybrid LLM 57.91 Smallest-LLM 42.53
Conditioning on user context helps, but
how it is modeled matters: Table 3 reports
persona-judge accuracy on the personalized
track. GMTRouter achieves the highest accuracy of 68.78, outperforming PersonalizedRouter (67.86) and the best user-agnostic
router, EloRouter (66.40). The strong performance of both personalized methods confirms the benefit of conditioning routing decisions on user context, while GMTRouter’s
additional 0.92-point gain over PersonalizedRouter shows that the way user context is
encoded and integrated remains important.
8
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
0.0 0.2 0.4 0.6 0.8
Cost weight β
kNNRouter
MLPRouter
MFRouter
GraphRouter
Hybrid LLM
SVMRouter
CausalLM
EloRouter
RouterDC
Smallest-LLM
Largest-LLM
4 2 2 2 1
6 4 4 3 3
7 5 3 4 4
2 1 7 8 5
9 10 10 9 7
3 6 5 5 6
8 3 1 1 2
10 7 8 6 8
1 8 6 7 10
11 11 11 11 9
5 9 9 10 11
Generic LLM Tasks
0.0 0.2 0.4 0.6 0.8
Cost weight β
2 1 1 1 3
11 3 4 3 1
10 2 2 4 2
9 5 6 5 6
5 4 3 7 8
1 5 7 5 6
3 7 10 8 7
4 5 7 5 6
8 9 5 2 4
6 8 9 7 8
7 6 8 6 5
Memory
0.0 0.2 0.4 0.6 0.8
Cost weight β
5 8 4 4 2
10 6 1 1 1
8 9 6 4 4
3 3 5 6 6
4 1 2 3 5
2 10 8 5 6
9 2 3 2 3
1 10 8 5 6
11 5 10 9 9
6 4 7 7 7
7 7 9 8 8
Vision
0.0 0.2 0.4 0.6 0.8
Cost weight β
5 3 5 5 4
3 2 4 4 3
5 1 1 1 1
2 4 2 3 8
6 5 3 2 2
4 6 6 7 7
8 9 10 11 6
1 7 7 8 8
9 11 9 6 5
7 8 8 9 9
9 10 11 10 10
TimeSeries
Figure 5: Router rankings across the Generic LLM Tasks, memory, vision, and time-series tracks
as the cost weight β increases. Each cell gives a router’s rank under the weighted performance–cost
objective, with smaller rank values indicating better performance.
5.3 PERFORMANCE–COST TRADE-OFFS
No single router is best under every performance–cost trade-off. Figure 5 ranks the routers by
reward within each category as the cost weight β grows, and the rankings shift dramatically along the
sweep. RouterDC tops Generic LLM Tasks when only quality matters, yet falls to tenth of eleven
under the most cost-sensitive setting; EloRouter leads Vision and TimeSeries at β “ 0 but drops out
of the lead once cost enters the objective. However, weakness at one operating point does not imply
weakness at another, as MLPRouter sits near the bottom of Vision under the quality-first setting yet
becomes the best choice there for every β ě 0.4. Therefore, it is practical to choose the router that
matches the performance–cost requirements of the deployment at hand.
10
−4
Inference cost (USD per query)
0.34
0.36
0.38
0.40
0.42
0.44
0.46
Performance
kNNRouter
SVMRouter
MLPRouter
MFRouter
EloRouter
Hybrid LLM
RouterDC
GraphRouter
Smallest-LLM
Largest-LLM
Figure 6: Performance–cost trade-offs of routers
averaged across the xRouteBench tracks. Each
point represents an operating setting with a different cost weight β, where higher performance and
lower per-query inference cost are preferred.
Increasing inference cost leads to improved
performance. Figure 6 presents the trade-off between performance and inference cost. For most
routers, performance and cost exhibit a clear
positive correlation, with the operating points
rising from the low-cost to the high-cost end.
This is because increasing the inference budget
unlocks more powerful and expensive models,
which perform better. Meanwhile, always calling the largest model incurs the highest cost yet
delivers only mediocre performance and is dominated by the learned routes, since many queries
that the largest model fails are solved by smaller
and cheaper ones. This confirms that no single
model covers all queries, which is exactly the
headroom that routing exploits.
5.4 ROUTING IN DEPLOYMENT:
REAL USERS AND MULTI-AGENT SYSTEMS
Table 4: Router performance on held-out real-user
sessions collected through the Slack deployment.
Accuracy measures how often each router’s model
selection agrees with the users’ pairwise preferences.
Router Acc. Router Acc.
PersonalizedRouter 83.05 RouterDC 65.25
EloRouter 82.20 kNNRouter 60.17
MLPRouter 78.81 kNN-MultiRound 60.17
SVMRouter 77.12 Smallest-LLM 55.08
Hybrid LLM 73.73 MFRouter 51.69
GMTRouter 70.70 Largest-LLM 41.53
GraphRouter 67.17 CausalLM 27.97
The deployment layer of LLMRouter carries
routers beyond static benchmarks. We study
two settings it enables: routing for real users
served through OpenClaw and routing inside
multi-agent systems.
Routing for real users. Using the OpenClaw
server of LLMRouter, we deploy the routing
stack behind Slack and collect live preference
feedback. 15 users contribute 40 sessions of
1 to 12 turns, totaling 234 pairwise records.
For each query, two models are sampled from
a pool of ten candidates, their answers are
9
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Planner
SubPlanne
r
SubPlanne
r
Planner
Actor
Actor
Planner
Sub-Planner
Actor
Actor
Actor
Actor Actor
Executor
(a) Star (b) Tree (c) Graph (d) Chain (e) Plan-Exec-Sum
Planner
Summarizer
Figure 7: Representative multi-agent system architectures and coordination topologies: (a)
star-based centralized coordination, (b) hierarchical tree-based delegation, (c) graph-based peer
interaction, (d) sequential chain collaboration, and (e) planner–executor–summarizer workflow.
Table 5: Router performance on the Generic LLM Tasks test split when each node in a multiagent system is routed independently. Results are reported across five coordination topologies,
with the final column showing the average performance.
Router Star Tree Graph Chain Plan-Exec-Sum Avg
Largest-LLM 69.00 67.00 77.20 69.00 75.20 71.48
kNNRouter 74.80 78.60 78.60 76.60 71.80 76.08
SVMRouter 76.20 75.60 80.00 74.40 75.20 76.28
MLPRouter 75.40 76.60 76.80 78.00 71.40 75.64
MFRouter 75.40 74.20 81.00 78.60 73.20 76.48
EloRouter 73.80 72.40 78.60 76.60 75.20 75.32
GraphRouter 68.20 70.80 66.20 72.00 69.00 69.24
RouterDC 77.60 79.60 74.20 72.00 76.20 75.92
shown in randomized positions, and the user marks one as better or declares a tie. We split by session
into 32 training sessions and 8 test sessions, train every router on the human training split, and score
how often its selection matches the human preference on held-out sessions. Table 4 shows that
PersonalizedRouter leads at 83.05, while the fine-tuned CausalLM router ranks last. The simulated
ranking does not fully transfer, as GMTRouter, the winner under the persona judge, drops to sixth on
real users, showing that it matters to validate personalized routers against real feedback.
Routing inside multi-agent systems. A multi-agent system (MAS) is conventionally instantiated
with a single base model shared by every agent. We instead treat model choice as a per-agent decision,
where a router receives the prompt of each functional node and selects the most suitable LLM for
that call. Since node prompts differ substantially, covering planning, execution, and verification
instructions, this setting stress-tests how well a router trained on ordinary queries generalizes.
Following MultiAgentBench (Zhu et al., 2025) and GraphPlanner (Feng et al., 2026), we instantiate
five coordination topologies, namely Star, Tree, Graph, Chain, and Plan-Exec-Sum, illustrated in
Figure 7 with node roles detailed in Appendix E. The learning-based routers are trained on the
Generic LLM Tasks training split, each test query is then fed through the full MAS, and the final
MAS answer is scored by the task metric. Table 5 shows that routing every node pays off, as six of
the seven learned routers beat always selecting the largest model on average, with MFRouter attaining
the best average of 76.48 against 71.48.
6 RELATED WORK
LLM Routing. Prior routers can be read along the axes of our formulation. Single-turn routers
differ mainly in how they encode a query and score candidates, from quality predictors that arbitrate
between a weak and a strong model (Ding et al., 2024; Ong et al., 2024) and classifiers over frozen
embeddings (Shnitzer et al., 2023; Stripelis et al., 2024; Li, 2025), to reward-guided rankings (Lu
et al., 2024a), prompt-conditioned preference models (Frick et al., 2025), contrastive query–model
matching (Chen et al., 2024), learned model embeddings (Zhuang et al., 2025), graph-based scorers
(Feng et al., 2024), and online methods under bandit feedback (Dai et al., 2024; Wang et al., 2025).
Multi-turn routers instead enrich the state across rounds, whether as cascades that escalate upon
failed verification (Chen et al., 2023; Aggarwal et al., 2024) or as agentic routers that decompose
a query and route sub-queries with reinforcement learning (Zhang et al., 2025; Feng et al., 2026).
10
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Personalized routers add the user to the state and learn from preference feedback (Xie et al., 2025; Dai
et al., 2025; Yu et al., 2026a). Each is developed in its own formalism and evaluated on its own stack;
LLMRouter instead expresses them as instantiations of a single sequential decision process behind
one interface, so single-turn, multi-turn, and personalized routers meet on the same performance–cost
frontier.
Routing Benchmarks and Evaluation. RouterBench (Hu et al., 2024) precomputes candidate
responses over a fixed pool, RouterEval (Huang et al., 2025b) aggregates large-scale performance
records for routing study, preference data from Chatbot Arena has served as routing supervision (Ong
et al., 2024), and recent benchmarks extend routing to vision–language pools (Huang et al., 2025a;
Ma et al., 2026). However, existing benchmarks each target a single scenario, one-shot text or image
QA, with fixed pools, and provide no pipeline for constructing supervision on new tasks or candidate
sets. LLMRouter closes this gap with automatic supervision construction and cost-aware evaluation
over configurable pools, and xRouteBench spans generic LLM tasks, memory-augmented, vision
(image and video), time-series, and personalized scenarios under one protocol.
7 CONCLUSION
We introduced LLMRouter, a unified framework for LLM routing that casts single-turn, multi-turn,
and personalized routing as instances of a common sequential decision process. LLMRouter also
provides an automatic pipeline for constructing routing supervision and evaluation for new tasks
and candidate pools, the multi-scenario xRouteBench benchmark, and an open-source library that
implements more than 16 routers behind a unified interface and supports deployment to real users
and multi-agent systems. We hope LLMRouter will serve as a common foundation for developing,
evaluating, and deploying LLM routers.
REFERENCES
Aaron Adcock, Aayushi Srivastava, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pande, Abhinav
Pandey, Abhinav Sharma, Abhishek Kadian, Abhishek Kumawat, Adam Kelsey, et al. The llama 4
herd: Architecture, training, evaluation, and deployment notes. arXiv preprint arXiv:2601.11659,
2026.
Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K
Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv
preprint arXiv:2508.10925, 2025.
Pranjal Aggarwal, Aman Madaan, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei
Zhou, Aditya Gupta, Dheeraj Rajagopal, Karthik Kappaganthu, Yiming Yang, et al. Automix:
Automatically mixing language models. Advances in Neural Information Processing Systems, 37:
131000–131034, 2024.
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan,
Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language
models. arXiv preprint arXiv:2108.07732, 2021.
Essential AI: Mike Callahan, Adarsh Chaluvaraju, Aleksa Gordic, Devaansh Gupta, Yash Jain, ´
Philip Monk, Michael Pust, Tim Romanski, Peter Rushton, Ali Shehper, Divya Shivaprasad,
Saurabh Srivastava, Anil Thomas, Alok Tripathy, Ameya Velingker, and Ashish Vaswani. Rnj1-5-Instruct, 2026. URL https://huggingface.co/EssentialAI/rnj-1-5-instruct. Longcontext Instruction-tuned model release.
Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use large language models while
reducing cost and improving performance. arXiv preprint arXiv:2305.05176, 2023.
Mark Chen. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374,
2021.
Shuhao Chen, Weisen Jiang, Baijiong Lin, James Kwok, and Yu Zhang. Routerdc: Query-based
router by dual contrastive learning for assembling large language models. Advances in Neural
Information Processing Systems, 37:66305–66328, 2024.
11
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina
Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of
the 2019 conference of the north American chapter of the association for computational linguistics:
Human language technologies, volume 1 (long and short papers), pp. 2924–2936, 2019.
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.
arXiv:1803.05457v1, 2018.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve
math word problems. arXiv preprint arXiv:2110.14168, 2021.
Xiangxiang Dai, Jin Li, Xutong Liu, Anqi Yu, and John Lui. Cost-effective online multi-llm selection
with versatile reward models. arXiv preprint arXiv:2405.16587, 2024.
Zhongjie Dai, Tao Feng, and Jiaxuan You. Personalizedrouter: Personalized llm routing via graphbased user preference modeling. arXiv preprint arXiv:2511.16883, 2025.
Deep Cogito. Cogito v2 preview: Deepseek 671b moe. https://huggingface.co/deepcogito/
cogito-v2-preview-deepseek-671B-MoE, 2025. Hugging Face model card. Accessed: 2026-
07-28.
Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Ruhle, Laks VS
Lakshmanan, and Ahmed Hassan Awadallah. Hybrid llm: Cost-efficient and quality-aware query
routing. arXiv preprint arXiv:2404.14618, 2024.
Tao Feng, Yanzhen Shen, and Jiaxuan You. Graphrouter: A graph-based router for llm selections.
arXiv preprint arXiv:2410.03834, 2024.
Tao Feng, Haozhen Zhang, Zijie Lei, Peixuan Han, and Jiaxuan You. Graphplanner: Graph memoryaugmented agentic routing for multi-agent llms. arXiv preprint arXiv:2604.23626, 2026.
Evan Frick, Connor Chen, Joseph Tennyson, Tianle Li, Wei-Lin Chiang, Anastasios N Angelopoulos,
and Ion Stoica. Prompt-to-leaderboard. arXiv preprint arXiv:2502.14855, 2025.
Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling synthetic data
creation with 1,000,000,000 personas. arXiv preprint arXiv:2406.20094, 2024.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models. arXiv preprint arXiv:2407.21783, 2024.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint
arXiv:2009.03300, 2020.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv
preprint arXiv:2103.03874, 2021.
Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt
Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-llm routing
system. arXiv preprint arXiv:2403.12031, 2024.
Zhehao Huang, Baijiong Lin, Jingyuan Zhang, Jingying Wang, Yuhang Liu, Ning Lu, Tao Li, and
Xiaolin Huang. Vl-routerbench: A benchmark for vision-language model routing. arXiv preprint
arXiv:2512.23562, 2025a.
Zhongzhan Huang, Guoming Ling, Yupei Lin, Yandong Chen, Shanshan Zhong, Hefeng Wu, and
Liang Lin. Routereval: A comprehensive benchmark for routing llms to explore model-level
scaling up in llms. arXiv preprint arXiv:2503.10657, 2025b.
12
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning.
Transactions on Machine Learning Research, 2022.
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot,
Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al.
Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris
Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al.
Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.
Yang Li. Rethinking predictive modeling for llm routing: When simple knn beats complex learned
routers. arXiv preprint arXiv:2505.12601, 2025.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint
arXiv:2412.19437, 2024.
Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, and Jingren Zhou.
Routing to the expert: Efficient reward-guided ensemble of large language models. In Proceedings
of the 2024 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 1964–1974, 2024a.
Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu.
Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning.
In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and
the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers),
pp. 6774–6786, 2021.
Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng,
Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning
of foundation models in visual contexts. In International Conference on Learning Representations,
volume 2024, pp. 23439–23554, 2024b.
Haoxuan Ma, Guannan Lai, and Han-Jia Ye. Mmr-bench: A comprehensive benchmark for multimodal llm routing. arXiv preprint arXiv:2601.17814, 2026.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei
Fang. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.
13851–13870, 2024.
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct
electricity? a new dataset for open book question answering. In Proceedings of the 2018 conference
on empirical methods in natural language processing, pp. 2381–2391, 2018.
Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez,
M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms with preference data. arXiv
preprint arXiv:2406.18665, 2024.
OpenClaw. OpenClaw: Personal AI assistant. https://github.com/openclaw/openclaw, 2026.
GitHub repository, accessed August 3, 2026.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for
machine comprehension of text. In Proceedings of the 2016 conference on empirical methods in
natural language processing, pp. 2383–2392, 2016.
Tal Shnitzer, Anthony Ou, Mírian Silva, Kate Soule, Yuekai Sun, Justin Solomon, Neil Thompson,
and Mikhail Yurochkin. Large language model routing with benchmark datasets. arXiv preprint
arXiv:2309.15789, 2023.
13
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari. Charadesego: A large-scale dataset of paired third and first person videos. arXiv preprint arXiv:1804.09626,
2018.
Dimitris Stripelis, Zhaozhuo Xu, Zijian Hu, Alay Dilipbhai Shah, Han Jin, Yuhang Yao, Jipeng
Zhang, Tong Zhang, Salman Avestimehr, and Chaoyang He. Tensoropera router: A multi-model
router for efficient llm inference. In Proceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing: Industry Track, pp. 452–462, 2024.
Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question
answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of
the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, 2019.
Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya
Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al.
Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118,
2024.
Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej,
Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical
report. arXiv preprint arXiv:2503.19786, 2025.
Xinyuan Wang, Yanchi Liu, Wei Cheng, Xujiang Zhao, Zhengzhang Chen, Wenchao Yu, Yanjie Fu,
and Haifeng Chen. Mixllm: Dynamic routing in mixed large language models. In Proceedings of
the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 10912–10922, 2025.
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming
Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi
Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: A more robust and challenging multi-task language
understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval:
Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813,
2024.
Encheng Xie, Yihang Sun, Tao Feng, and Jiaxuan You. Gmtrouter: Personalized llm router over
multi-turn user interactions. arXiv preprint arXiv:2511.08590, 2025.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint
arXiv:2412.15115, 2024.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388,
2025.
Fangxu Yu, Tao Feng, Dehai Min, Lu Cheng, Ge Liu, and Tianyi Zhou. Tsrouter: Dynamic modalitymodel selection for time series reasoning. arXiv preprint arXiv:2607.08940, 2026a.
Fangxu Yu, Xingang Guo, Lingzhi Yuan, Haoqiang Kang, Hongyu Zhao, Lianhui Qin, Furong
Huang, Bin Hu, and Tianyi Zhou. Tsrbench: A comprehensive multi-task multi-modal time series
reasoning benchmark for generalist models. arXiv preprint arXiv:2601.18744, 2026b.
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine
really finish your sentence? In Proceedings of the 57th annual meeting of the association for
computational linguistics, pp. 4791–4800, 2019.
Haozhen Zhang, Tao Feng, and Jiaxuan You. Router-r1: Teaching llms multi-round routing and aggregation via reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information
Processing Systems, 2025.
14
LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.
Kunlun Zhu, Hongyi Du, Zhaochen Hong, Xiaocheng Yang, Shuyi Guo, Daisy Zhe Wang, Zhenhailong Wang, Cheng Qian, Robert Tang, Heng Ji, et al. Multiagentbench: Evaluating the collaboration
and competition of llm agents. In Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp. 8580–8622, 2025.
Richard Zhuang, Tianhao Wu, Zhaojin Wen, Andrew Li, Jiantao Jiao, and Kannan Ramchandran. Embedllm: Learning compact representations of large language models. In International Conference
on Learning Representations, volume 2025, pp. 76913–76926, 2025.