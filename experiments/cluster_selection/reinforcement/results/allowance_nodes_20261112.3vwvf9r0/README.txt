Per-puzzle DFS nodes across structural cluster allowances

Source: allowance_plateau_20261030.bih66zeb
Uses only the existing 23-puzzle validation sweep; no solver runs or training.
These are the previous sweep's puzzle indices, not the newer 55-puzzle evaluation.
The sweep uses nested structural proposal pools and the original branching rule.
It does not use the newly added oldest-pruned tie-break.
1 DFS node includes the root and means solved without branching.
Rows are sorted by maximum available completed node count, descending, then index.
Timeouts have no numeric node value and are excluded from the colour scale.

Guards: 600 seconds per complete solve; 100000 DFS nodes, independently.
The wall timer includes cluster generation, contraction, GAC and DFS.
It is set once around attachment and solve; network/state setup is measured separately.
The timeout handler does not save the partial DFS count.

Allowance | All 23 total | Puzzle 113 nodes | Puzzle 113 seconds | Other 22 total
32 | 188 | 136 | 18.537 | 52
64 | 173 | 129 | 40.216 | 44
128 | 155 | 112 | 83.280 | 43
256 | 52 | 7 | 20.666 | 45
512 | 66 | 22 | 223.198 | 44
1024 | incomplete | timeout | 600.001 | 36

Source JSONL SHA-256:
heuristic_32.jsonl: 873a43f39176db4e5fa0a7747de267fb5c4b5be6aed261edeab8c7645eabd3ff
heuristic_64.jsonl: a416f76f0bdc7a3d6a564a1a446d2adc247ad9ff45fbb6e1906add1e38400ea4
heuristic_128.jsonl: 2331d08c367b175366015c935ac8b49aeeec7b271aa2831450828be921af1ecb
heuristic_256.jsonl: bb780d544b3a950498c50a46ca96c3f42a7f2b55c9e008391ebc650242c6fcbc
heuristic_512.jsonl: 721f2f380e8aeb796a49c1deb31ce5b154d8b5f40b3e7038c201d8da41c4b079
heuristic_1024.jsonl: 3700e53eeb0c5b2107aa5ea4db7df33b11aab9865f4b0e246e94ad450621fad5
