# IMSA West Checkers AI — Student Experience

Status: game-first production interface
Experiment type: engineering integration
Reviewed: 2026-08-01

## Objective

Make the real trained policy immediately playable for IMSA West students, then explain the project without turning the site into a course, storybook, dashboard, or marketing page.

The falsifiable release objective is: a first-time visitor can start a real match in one obvious interaction, identify legal pieces and destinations, complete a move with pointer, touch, keyboard board, or standard buttons, and accurately state that self-play produced experience while PPO updated a policy that scores legal actions.

## Page structure

The public site is one focused page:

1. IMSA West identity and a direct game invitation.
2. Side choice, Start button, live board, turn state, contrast option, and move history.
3. Four concise learning steps: encode the board, self-play, PPO update, checkpoint testing.
4. Authentic update-4,608 evidence and its limitations.
5. A short teacher note distinguishing the neural policy from the rules engine.

There are no generated scene images, fictional characters, cinematic transitions, route progress, theme system, or lesson simulations in the shipped application. Historical artwork is retained outside `public/` only as recoverable repository provenance.

## Authenticity contract

- The Python server—not React—owns legality, mandatory captures, multi-jumps, promotion, terminal conditions, and policy inference.
- The UI always requests deterministic greedy action selection to avoid exposing unnecessary setup choices.
- The network receives eight actor-canonical `8 × 8` planes and a 128-slot legal-action mask.
- The selected deployment is update 4,608 with 470,410 parameters.
- The selected update had trained through 37,748,736 self-play transitions; the full 6,144-update practice run later reached 50,331,648.
- Recorded fixed-protocol results are 432 wins, 0 draws, 0 losses against random and 354 wins, 70 draws, 8 losses against the project Minimax-2 proxy.
- Those results are checkpoint-selection evidence, not a human rating or sealed-test claim.
- The interface never says the model thinks, knows, wants, feels, or understands.

## Accessibility and privacy

- The board supports pointer, touch, roving keyboard focus, and equivalent legal-move buttons.
- Selection, movable pieces, destinations, last move, kings, and color identity use text or shape in addition to color.
- A stronger-contrast board option is visible next to play.
- The page has semantic landmarks, a skip link, visible focus, reduced-motion handling, responsive layouts, and live turn/error announcements.
- The application has no accounts, analytics, grading, chat, advertising, or student-data collection.
