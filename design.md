# RotoStream design system

The editor is a post-production workbench for isolating a subject, propagating
its mask across a clip, and preparing a deliverable. This system keeps the
viewer and its controls at the centre of the experience.

## Genre and structure

- Genre: modern-minimal, tuned for an image-led editing tool.
- App structure: three working zones — clip library, viewer/timeline, contextual
  inspector — with a compact workflow across Prompt, Track, and Export.
- Login and editor share the same wordmark, type, surfaces, and signal colours.

## Theme: RotoStream Cut

A custom deep-slate palette preserves the API's exact sky-blue mask colour. The
canvas is the quietest surface; the inspector and library rise one step above it.
Borders establish grouping. Colour marks selection, mask state, and action.

- Canvas: `#0b0e12`; shell: `#11161b`; raised surface: `#1a2229`.
- Text: `#f2f5f7`; secondary: `#b5c0c8`; muted: `#87949e`.
- Accent: `#38bdf8` (must match `api/app/masks.py::ACCENT`).
- Positive: `#4ade80`; negative: `#fb7185`; warning: `#fbbf24`.
- Display and body: IBM Plex Sans; timecodes and measured values: IBM Plex Mono.

## Spacing, shape, and motion

- Four-point spacing scale lives in `web/src/app/tokens.css`.
- Panel radius: 8px; controls: 6px; pills are reserved for compact statuses.
- Motion is functional and brief: 120–180ms for focus, hover, and active states.
  No ambient loops or animated page reveals. Respect `prefers-reduced-motion`.
- Primary actions use a filled accent treatment; secondary actions use a quiet
  border or text treatment. Mask colour remains consistent between controls and
  rendered overlays.

## Interface rules

- The viewer receives the most space. Library metadata and technical health stay
  compact; advanced memory details remain available without competing with the
  editing sequence.
- Prompts → Track → Export is the default working order. Disabled actions explain
  what is needed to enable them.
- Mono type is for timecode, dimensions, frame counts, and technical values, not
  general interface copy.
- Every editing control has keyboard focus, an accessible name, and a visible
  state. Mobile layouts keep the clip shelf and workflow usable without hover.
