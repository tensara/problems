---
slug: "scaled-dot-attention"
title: "Scaled Dot-Product Attention"
difficulty: "HARD" 
author: "sarthak"
tags: ["attention"]
parameters:
  - name: "Q"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "K" 
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "V" 
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "B"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "H" 
    type: "size_t"
    pointer: "false"
    constant: "false"
  
  - name: "S"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "E" 
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Implement scaled dot-product attention, a key component of transformer architectures:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where Q, K, and V are the query, key, and value matrices, and d_k is the dimension of the key vectors (embed_dim).

## Input:
- Matrix $Q$ of size $(\text{B} \times \text{H} \times \text{S} \times \text{E})$ (query)
- Matrix $K$ of size $(\text{B} \times \text{H} \times \text{S} \times \text{E})$ (key)
- Matrix $V$ of size $(\text{B} \times \text{H} \times \text{S} \times \text{E})$ (value)

## Output:
- Matrix $\text{output}$ of size $(\text{B} \times \text{H} \times \text{S} \times \text{E})$

## Notes:
- Input matrices are stored in a row-major format
- The scaling factor is $\sqrt{\text{E}}$
- For this problem, we can assume the embedding dimension is the same for query, key and value matrices ($\text{E}$)
- Similarly, the number of heads is the same for query, key and value matrices ($\text{H}$)
