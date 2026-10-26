# Clinical UI Design System & Style Tokens (PACS / DICOM Standard)

> **Design Objective:**  
> A radiologist or thoracic clinician must recognize this interface within 5 seconds as professional, regulated clinical workstation software (akin to OHIF Viewer, GE Centricity, Philips IntelliSpace, or Siemens syngo.via), **not** a startup tech dashboard.

---

## 1. Core Visual Directives

1. **Dark Neutral Core:** Near-black background (`#0e1013`) with flat, low-contrast slate panels (`#16191d`, `#1c2026`).
2. **Strict Elimination of Tech/Startup Clichés:**
   - **Zero** gradients, glowing borders, neon cyan/purple accents, glassmorphism, background blurs, hero banners, floating decorative badges, or marketing text.
   - **Zero** emojis in the UI, code, or documentation.
   - **Zero** rounded cards ($> 4\text{px}$). All containers use flat surfaces with thin $1\text{px}$ borders and $2\text{px}-4\text{px}$ border radii.
3. **Information Density & Spacing Grid:**
   - 4px / 8px spacing grid.
   - Dense, high-legibility layouts with zero wasted padding.
   - Tabular numerals (`font-variant-numeric: tabular-nums`) for all measurements, coordinates, volume values, and probabilities.
   - Explicit clinical units on every number: $\text{mm}$, $\text{mm}^3$, $\text{HU}$, $\%$, $\text{days}$.
4. **Functional Semantic Color Only:**
   - **Accent / Selection:** Slate/Clinical Blue (`#3b82c4`) — indicates active tool, selected row, or focused viewport.
   - **Warning / Review Closely:** Amber (`#f59e0b`) — indicates high uncertainty, border-line risk, or out-of-distribution scan parameters.
   - **Action Required / Critical Finding:** Red (`#ef4444`) — reserved strictly for suspicious nodules, critical triage priority, or scan corruption.
   - **Safe State:** Neutral grey (`#94a3b8`) or subtle muted slate. Green is **never** used to imply "cancer-free" or "safe" (to avoid false reassurance).
   - **Contrast Compliance:** All text and glyph combinations satisfy WCAG AA contrast standards ($\ge 4.5:1$).

---

## 2. Design Tokens

### 2.1 Color Palette
```css
:root {
  /* Surface Layers (PACS Dark Neutral) */
  --bg-app:       #0e1013; /* Primary canvas & viewport background */
  --bg-panel:     #16191d; /* Sidebars, headers, tool docks */
  --bg-surface:   #1c2026; /* Table headers, card panels, active tabs */
  --bg-hover:     #242932; /* Interactive row hover */
  --bg-active:    #2d3440; /* Selected finding row / active tool */

  /* Borders & Dividers */
  --border-color: #262b33; /* 1px neutral container boundaries */
  --border-subtle:#1f232a; /* Secondary table row dividers */
  --border-focus: #3b82c4; /* Keyboard & accessibility focus rings */

  /* Clinical Meaning & State */
  --accent-color: #3b82c4; /* PACS blue for crosshairs and active tools */
  --accent-subtle:#142232; /* Light tinted background for selected rows */
  --status-amber: #f59e0b; /* Amber for "Needs Review / Moderate-High Uncertainty" */
  --status-amber-bg: #2b200b;
  --status-red:   #ef4444; /* Red for "Suspicious Finding / High Triage Priority" */
  --status-red-bg: #2d1313;
  --status-neutral:#94a3b8; /* Routine priority / benign appearance */

  /* Typography Colors */
  --text-primary:  #e2e8f0; /* Crisp white-slate for values & labels */
  --text-secondary:#94a3b8; /* Muted slate for metadata tags */
  --text-muted:    #64748b; /* Low-emphasis units, timestamps */
  --text-dim:      #475569; /* Table borders, inactive icons */

  /* Typography Families */
  --font-sans: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', Consolas, 'Courier New', monospace;

  /* Metric Radii */
  --radius-xs: 2px;
  --radius-sm: 3px;
  --radius-md: 4px;
}
```

### 2.2 Typography Scale
| Role | Size | Weight | Line Height | Usage |
|---|---|---|---|---|
| Caption / Corner DICOM | 10px | 500 / 600 | 12px | Viewport overlay text, orientation markers, scale bar |
| Compact Metadata | 11px | 400 / 500 | 14px | Table cells, timestamps, badge labels |
| Body / Standard UI | 12px | 400 | 16px | Field labels, dialog text, navigation labels |
| Section Subheading | 13px | 600 | 18px | Panel subheaders, finding group banners |
| Panel Header | 14px | 600 | 20px | Main panel titles, patient banner header |
| Large Numeric Readout | 18px | 600 (tabular) | 22px | Calibrated probability %, Nodule diameter mm |

---

## 3. Mandatory DICOM Viewport Overlay Layout

The primary 3D / CT viewport must project traditional PACS corner overlays in standard monospaced text with text shadows for legibility against diverse lung HU values:

```
┌────────────────────────────────────────────────────────────────────────┐
│ [PAT-0087-LIDC]                               [Series: 3D_AXIAL_1MM]   │
│ Study: 2026-09-22 10:42                       Thickness: 1.00 mm       │
│ Modality: CT CHEST LDCT                       Spacing: 1.0 x 1.0 mm    │
│                                                                        │
│                              [ A ]                                     │
│                                                                        │
│                                                                        │
│    [ R ]                                                 [ L ]         │
│                                                                        │
│                                                                        │
│                              [ P ]                                     │
│                                                                        │
│ Slice: 92 / 196                               Zoom: 100%               │
│ W: 1500  L: -600 (LUNG)                       Z: -128.0 mm             │
│ Field: 350.0 mm                               [=== Scale: 5 cm ===]    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Triage Worklist Layout Specification

The worklist screen (`WorkstationPanel`) implements a dense, sortable RIS-style reading queue:
- **Columns:** Priority Flag (`URGENT`, `HIGH`, `ROUTINE`, `DEFERRED`), Study Date, Patient ID, Age/Sex, Scan Modality, Calibrated Malignancy Risk %, Epistemic Uncertainty Band, Deferral Status, Assigned Radiologist.
- **Sorting:** Multi-column sorting by priority score, risk, uncertainty, or acquisition date.
- **Keyboard Navigation:**
  - `↑` / `↓`: Move selection up/down.
  - `Enter`: Open study in 2x2 CT Viewer.
  - `Space`: Toggle Blinded-First review mode.

---

## 5. Audit & Compliance Indicators

1. **Persistent Regulatory Header:** Every view must feature a permanent, un-dismissible top notification strip:
   `RESEARCH USE ONLY — Investigational Decision Support Tool — Not Approved for Autonomous Diagnostic Decisions`
2. **Synthetic Data Badge:** Whenever demo, phantom, or synthetic training cohorts are rendered, a high-visibility badge reading `SYNTHETIC DEMO DATA` must be pinned to the viewport banner.
3. **No Fabricated Numbers:** If a metric (e.g. historical scan or uncertainty CI) has not been computed from a verified run, it must display `Not available` or `TBD (not yet run)` instead of fallback approximations.
