# self-forcing — a fast video model trained on its own frames

> **▶ [Open this demo](index.html)** · [all demos →](../index.html)

Frame `n` under teacher forcing carries `n × step` error. Every self-forced frame carries `step`.

Frame 1 matches. Later frames drift only on the teacher-forced train. A step of 0 leaves both flat.
