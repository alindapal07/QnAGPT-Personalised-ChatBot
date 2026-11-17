# QnAGPT — Design System & UI Architecture Specification

**Product:** QnAGPT — Premium AI Workspace  
**Theme:** Light Professional SaaS Workspace  
**Primary Aesthetics Inspiration:** Clean Modern Conversational Canvas (`media_1794908342691_d7a840c6.jpg`)

---

## 1. Design Tokens & Color Palette

The QnAGPT design system enforces a calm, highly legible, professional palette centered around a soft airy periwinkle-blue application background, pure white elevated surfaces, restrained indigo-purple primary accents, and crisp semantic status signals.

```css
:root {
    /* Backgrounds & Canvas */
    --canvas-background:   #EEF3FB; /* Soft airy blue-grey canvas */
    --app-background:      #FFFFFF; /* White card surface */
    --main-surface:        #FFFFFF; /* Primary workspace surface */
    --sidebar-background:  #FFFFFF; /* Elevated sidebar surface */
    --surface-secondary:   #F7F8FC; /* Subtle recessed backgrounds */
    --surface-hover:       #F0F3FA; /* Hover state tint */

    /* Typography & Text */
    --text-primary:        #202333; /* Deep neutral dark text */
    --text-secondary:      #73798C; /* Balanced supporting text */
    --text-muted:          #A0A5B4; /* De-emphasized placeholder & meta */

    /* Borders & Dividers */
    --border:              #E9EBF2; /* Soft hairline border */
    --border-strong:       #D3D8E6; /* Focus & active border */

    /* Brand Accent & Highlights */
    --accent:              #5B61F6; /* Modern indigo-purple primary */
    --accent-hover:        #4B50E8; /* Darker primary hover */
    --accent-soft:         #EEF2FF; /* Active conversation pill background */
    --accent-border:       #D5DAFC; /* Active conversation pill outline */

    /* Semantic Status Signals */
    --success-fg:          #239B72; /* Emerald success */
    --success-bg:          #EBF8F3; 
    --success-border:      #B7E7D5;
    --warn-fg:             #D99A35; /* Warm amber warning */
    --warn-bg:             #FEF7EC;
    --warn-border:         #F7D8A7;
    --err-fg:              #D95363; /* Ruby red error */
    --err-bg:              #FDF2F3;
    --err-border:          #F7C1C7;

    /* Radii & Elevation */
    --radius-sm:           8px;
    --radius-md:           12px;
    --radius-lg:           16px;
    --radius-xl:           20px;
    --radius-pill:         999px;
    --shadow-sm:           0 1px 2px rgba(32, 35, 51, 0.04);
    --shadow-card:         0 4px 20px rgba(32, 35, 51, 0.05);
    --shadow-float:        0 8px 30px rgba(32, 35, 51, 0.08);
}
```

---

## 2. Typography Hierarchy

The interface utilizes a clean sans-serif stack headed by **Inter**, falling back cleanly to system UI font stacks (`system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`).

| Element | Size | Weight | Line Height | Letter Spacing | Color |
|---|---|---|---|---|---|
| **Brand Wordmark** | `16px` | `700` (Bold) | `1.2` | `-0.02em` | `#202333` |
| **Workspace Page Title** | `22px` | `700` (Bold) | `1.3` | `-0.025em` | `#202333` |
| **Welcome Heading** | `24px` | `700` (Bold) | `1.3` | `-0.025em` | `#202333` |
| **Section Header** | `14px` | `600` (Semi-bold) | `1.4` | `-0.01em` | `#202333` |
| **Category Label** | `11px` | `600` (Semi-bold) | `1.2` | `0.06em` (Uppercase) | `#73798C` |
| **Body Paragraph** | `14.5px`| `400` (Regular) | `1.65` | `normal` | `#202333` |
| **Conversation Item** | `13px` | `500` (Medium) | `1.3` | `normal` | `#202333` |
| **Metadata & Captions** | `12px` | `400` (Regular) | `1.4` | `normal` | `#73798C` |
| **Inline Code / Tokens** | `12.5px`| `500` (Medium) | `1.4` | `normal` | `#202333` |

---

## 3. Layout Architecture & Spatial Rules

### 3.1 Dual-Surface Canvas
The layout is designed around two floating white surfaces against the `--canvas-background` (`#EEF3FB`):
1. **Left Sidebar Card:** Fixed or collapsible white card with 20px radius, 1px border, and soft elevation (`--shadow-card`).
2. **Main Workspace Card:** Expansive white card with 20px radius, generous internal padding (`1.4rem 2rem 6.5rem`), housing the conversation canvas or workspace views.

### 3.2 Desktop Layout (Width >= 769px)
- **Sidebar Width:** 280px fixed width, independent scrollable conversation container (height: 260px).
- **Workspace Max Width:** 1140px centered (expands to 1260px when sidebar collapsed; constrained to 880px on Chat for comfortable line reading length).
- **Composer Clearance:** 6.5rem bottom padding on `.block-container` ensures messages never hide behind the floating composer.

### 3.3 Mobile & Tablet Responsive Rules
- **Tablet (<= 1024px):** Sidebar remains visible; container padding reduces to `1rem 1.25rem`.
- **Mobile (<= 768px):** Sidebar converts into an off-canvas drawer with full height, modal shadow, and slide-in animation. Main container radius adjusts to 12px with zero outer margins.

---

## 4. Component Inventory

### 4.1 Primary Action Button (`+ New chat`)
- **Pill Shape:** Fully rounded (`border-radius: 999px`).
- **Gradient Background:** `linear-gradient(135deg, #5B61F6 0%, #4B50E8 100%)`.
- **Height & Typography:** `38px` height, `13.5px` semi-bold white text with subtle shadow (`0 4px 14px rgba(91, 97, 246, 0.28)`).
- **Hover:** Darker purple gradient with gentle translateY lift (`-1px`).

### 4.2 Conversation List Items
- **Inactive Items:** Transparent background, `10px` radius, hover tint `#F0F3FA`, truncated with ellipsis.
- **Active / Selected Item:** Lavender pill background (`#EEF2FF`), `1px solid #D5DAFC`, bold text `#5B61F6`, with action popover menu and active dot indicator (`#5B61F6`).

### 4.3 Floating Message Composer
- **Pill Container:** Centered at the viewport bottom, `border-radius: 999px`, white background (`#FFFFFF`), `1px solid #E9EBF2`, soft floating elevation (`0 8px 30px rgba(32, 35, 51, 0.08)`).
- **Input Field:** Borderless multiline text area with placeholder `"What's in your mind?... (Enter to send, attach files with +)"`.
- **Send Button:** Circular 36px button in vibrant `#5B61F6` with paper airplane SVG icon.

### 4.4 Assistant Turn & Action Footer
- **Identity Header:** `QnAGPT` wordmark paired with an assistant sparkle verified SVG badge.
- **Content:** Markdown rendered cleanly without card borders, supporting numbered lists, tables, syntax-highlighted code blocks, and source citations.
- **Action Toolbar:** 
  - Helpful (Thumbs Up)
  - Not Helpful (Thumbs Down)
  - Copy Response (with copy feedback transition)
  - More Options (Export Markdown)
  - Dedicated "Regenerate" Button on right with refresh icon.

### 4.5 Welcome Screen Prompt Pills
- **Headline:** *"What would you like to explore today?"*
- **Sub-headline:** *"Ask questions, explore your documents, or let an AI agent help you get things done."*
- **Categorized Starter Cards:** 2-column grid of subtle cards with Material icons covering concepts, document summarization, research analysis, knowledge base search, and problem solving.

### 4.6 Metric Strips & Status Pills
- **Metric Card:** Clean rectangular surface (`border-radius: 12px`, `padding: 10px 14px`) with uppercase muted label and bold tabular numeral value.
- **Semantic Status Pills:** Fully rounded badges with color-matched background, border, and text (`.status-pill-ok`, `.status-pill-warn`, `.status-pill-err`, `.status-pill-accent`).

---

## 5. Interaction States & Transitions

- **Hover Transitions:** All buttons, cards, and interactive elements transition smoothly over `0.12s–0.15s ease`.
- **Focus Rings:** Text inputs and interactive controls exhibit a 2px soft accent glow (`box-shadow: 0 0 0 2px #EEF2FF; border-color: #5B61F6;`).
- **Active State:** Instant visual feedback without unstyled flashes or layout shifts during Streamlit reruns.
- **Loading & Spinners:** Minimal, non-blocking Streamlit spinner states that maintain the integrity of existing canvas contents.
