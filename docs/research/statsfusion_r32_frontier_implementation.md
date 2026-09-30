# StatsFusion frontier implementation

The frontier run is registered in `configs/hierarchical_v4_r32_deep_frontier.yaml`.
It keeps the existing single-seed Deep-only protocol and enables three optional modules:

- learned-query masked attention pooling in the proposal verifier;
- zero-initialized FiLM conditioning of the boundary endpoint network with proposal metadata;
- a masked temporal contrastive regularizer on adjacent state hidden representations.

The default configurations leave all three modules disabled. The frontier configuration must
use a new run name and `--fresh`; checkpoints made with the previous pooling or endpoint input
dimensions are not compatible. The results remain development/stress evidence until a complete
outer evaluation and raw-session replay pass.
