---
slug: "array-sort"
title: "Array Sorting"
difficulty: "EASY"
author: "sarthak"
tags: ["sorting"]
parameters:
  - name: "a"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "b" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n"
    type: "size_t"
    pointer: "false"
    constant: "false"

---

Sort an array of floating-point numbers in ascending order.

## Input:
- Array $a$ of floating-point numbers of size $n$

## Output:
- The same array $a$ sorted in ascending order stored in $b$

## Notes:
- The input array $a$ is stored in row-major order in memory
- Array values are floating-point numbers in the range [0, 1000]