# Checkers Web Reference Study

## Inspected source

- Repository: [`nablsi14/react-checkers`](https://github.com/nablsi14/react-checkers)
- Revision: [`72d05046555d91c1a50bc7a4e60f30e3e600ec68`](https://github.com/nablsi14/react-checkers/tree/72d05046555d91c1a50bc7a4e60f30e3e600ec68)
- License: [MIT](https://github.com/nablsi14/react-checkers/blob/72d05046555d91c1a50bc7a4e60f30e3e600ec68/LICENSE), copyright Nathaniel Sigafoos (2018)
- Inspection date: 2026-07-29
- Files inspected: `Board.tsx`, `Square.tsx`, `Piece.tsx`, `Board.css`, `GameContainer.tsx`, `BoardMenu.tsx`, and `BoardMenu.css`.

## Clean-room observations

The reference keeps board, square, and piece rendering in separate React components, uses selection as explicit UI state, groups controls outside the board, changes the frame to indicate turn, and briefly stages AI movement instead of treating it as an invisible state update. These are interaction concepts, not copied implementation.

The local harness will independently use semantic button cells, a CSS grid that scales with the viewport, explicit legal-destination and last-move states, a nearby game-control panel, and a visible “model thinking” state. Unlike the reference, clicking a valid destination immediately submits the move because the existing Python engine already returns an exact legal action map.

## Deliberately not reused

- No source code, class names, layout measurements, selectors, algorithms, dependencies, or state structures were copied.
- No `black_piece.png`, `red_piece.png`, `crown.png`, favicon, or other bundled art was copied.
- No client-side rule or AI implementation was reused. The repository's existing `CheckersEnv`, move generator, notation, action encoding, and trained `PolicyAgent` remain authoritative.
- Local-storage games, editable player names, score calculation, modal implementation, routing, and the reference's explicit “Make move” interaction are outside this harness's frozen scope.

The result therefore draws only on ordinary interface patterns observed in a permissively licensed project and remains a new implementation tied to this repository's tested rules engine.
