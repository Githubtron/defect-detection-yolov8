# Aircraft Defect System ✈️

A modern web application for tracking, reporting, and managing aircraft defects and maintenance issues. Built as a high-performance single-page application with a polished, themeable UI.

> **Note for report readers:** this document describes the application's purpose, the technology stack it is built upon, the features currently implemented, and how to run and extend the project.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Available Scripts](#available-scripts)
- [Design & Theming](#design--theming)
- [Roadmap](#roadmap)

---

## Overview

The **Aircraft Defect System** is a front-end application designed to streamline the process of recording, categorizing, and resolving defects found on aircraft during inspections, operations, and maintenance. It aims to replace paper-based defect logs with a fast, searchable, and visually clear digital workflow.

The application is currently in its foundation phase: the tooling, design system, theme engine, and reusable visual components are in place, ready for the domain features (defect reporting, fleet tracking, dashboards) to be layered on top.

---

## Features

### Implemented

| Feature | Description |
| --- | --- |
| **Design System** | shadcn/ui + Radix UI component library wired with Tailwind CSS v4 design tokens (light & dark color palettes, radius scale, chart colors, sidebar colors). |
| **Theme Engine** | Full light / dark / system theme support. Persists the user's choice in `localStorage`, listens to OS preference changes, disables transition flicker on switch, and even syncs across browser tabs. Press **`D`** to toggle dark mode from anywhere. |
| **Animated Aurora Background** | A real-time WebGL background rendered with `ogl` — procedural simplex noise generates flowing aurora bands with configurable colors, amplitude, and blend. GPU-accelerated and fully responsive. |
| **Iconography** | HugeIcons icon library integrated with the component setup. |
| **Typography** | Inter Variable font (`@fontsource-variable/inter`) with a dedicated heading/sans font stack. |
| **Path Aliasing** | `@/*` alias maps to `src/*` for clean, consistent imports. |

### Planned (Roadmap)

- Defect creation, categorization, and status tracking
- Search / filter over defect records
- Fleet & aircraft registration management
- Dashboard with charts and KPIs
- Export / reporting of defect data

---

## Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| **Language** | [TypeScript](https://www.typescriptlang.org/) (~6, strict mode) | Type-safe application code |
| **UI Library** | [React](https://react.dev/) 19 | Component-based user interface |
| **Build Tool** | [Vite](https://vite.dev/) 8 | Fast dev server & production bundling |
| **Styling** | [Tailwind CSS](https://tailwindcss.com/) v4 | Utility-first styling with CSS-first config |
| **Components** | [shadcn/ui](https://ui.shadcn.com/) + [Radix UI](https://www.radix-ui.com/) | Accessible, themeable UI primitives |
| **WebGL** | [ogl](https://github.com/oframe/ogl) | GPU-accelerated aurora background effect |
| **Icons** | [HugeIcons](https://hugeicons.com/) | Icon set used across the UI |
| **Animations** | [tw-animate-css](https://github.com/vercel/tw-animate-css) | Tailwind-compatible animation utilities |
| **Utilities** | clsx + tailwind-merge (`cn()`) | Conditional class composition |
| **Linting** | ESLint 10 + typescript-eslint | Static analysis |
| **Formatting** | Prettier 3 + `prettier-plugin-tailwindcss` | Consistent code style |

### Why this stack

- **React + TypeScript** gives a strongly typed, component-driven architecture that scales as defect-management workflows grow in complexity.
- **Vite** provides near-instant hot reload and an optimized production build.
- **Tailwind CSS v4 + shadcn/ui** delivers a consistent, accessible design system with minimal boilerplate, and full theming via CSS variables (light/dark).
- **ogl** adds a distinctive, low-cost visual identity (the aurora backdrop) without pulling in a heavy 3D engine.

---

## Project Structure

```
aircraft-defect-system/
├── src/
│   ├── components/
│   │   ├── Aurora.tsx          # WebGL aurora background (simplex noise, ogl)
│   │   ├── theme-provider.tsx  # Theme context (light/dark/system) + D-key toggle
│   │   └── ui/                 # shadcn/ui components
│   ├── lib/
│   │   └── utils.ts            # cn() class-merge helper
│   ├── App.tsx                 # Root application component
│   ├── index.css               # Tailwind import + design tokens (light/dark)
│   └── main.tsx                # Entry point (wraps app in ThemeProvider)
├── components.json             # shadcn/ui configuration
├── vite.config.ts              # Vite config + @ path alias
├── tsconfig.json / tsconfig.app.json
├── eslint.config.js
├── package.json
└── README.md
```

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 20.19+ (or 22.12+)
- npm (bundled with Node.js)

### Installation

```bash
# 1. Install dependencies
npm install

# 2. Start the dev server
npm run dev
```

Open the printed local URL (typically `http://localhost:5173`) in your browser.

### Production build

```bash
npm run build      # type-checks (tsc -b) then bundles with Vite
npm run preview    # serve the production build locally
```

---

## Available Scripts

| Script | Command | Description |
| --- | --- | --- |
| `dev` | `vite` | Start the development server with hot reload |
| `build` | `tsc -b && vite build` | Type-check the project and create a production bundle |
| `preview` | `vite preview` | Preview the production build locally |
| `lint` | `eslint .` | Run ESLint over the codebase |
| `typecheck` | `tsc --noEmit` | Run the TypeScript compiler without emitting files |
| `format` | `prettier --write "**/*.{ts,tsx}"` | Auto-format all source files |

---

## Design & Theming

- **Design tokens** live in `src/index.css` as CSS variables under `:root` (light) and `.dark` (dark), including background, foreground, primary, muted, border, ring, chart, and sidebar colors — all defined in the OKLCH color space for perceptual consistency.
- **Theme resolution** (`theme-provider.tsx`): theme can be `light`, `dark`, or `system`. When `system` is selected, the provider listens to `prefers-color-scheme` and updates live. Transitions are temporarily disabled during switches to avoid flash.
- **Keyboard shortcut**: press `D` to toggle dark/light mode (ignored while typing in inputs).
- **Aurora background**: the `Aurora` component accepts `colorStops`, `amplitude`, `blend`, `time`, and `speed` props, letting the backdrop be tuned per-screen.

---

## Roadmap

1. Defect CRUD: create, edit, categorize, and resolve aircraft defects
2. Search, filtering, and sorting of defect records
3. Aircraft & fleet registration
4. Analytics dashboard (charts from the existing chart token palette)
5. Data export (CSV/PDF) for reporting

---

## License

Private project — all rights reserved.
