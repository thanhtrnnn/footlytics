---
name: footlytics web
description: "Observed visual system for the football film programme and technical showcase"
colors:
  paper: "#f2f1e9"
  white: "#fff"
  ink: "#142219"
  muted: "#4e5c52"
  rule: "#b9c1b6"
  accent: "#365b20"
  field: "#15372d"
  pitch-line: "#c8dccb"
  team-home: "#e5f06c"
  team-away: "#80c9cc"
  panel: "#e2e5d9"
  chapter: "#e8ebdf"
  on-dark-muted: "#c0d1c3"
  on-dark-copy: "#d5e1d5"
  dark-rule: "#547360"
typography:
  display:
    fontFamily: '"Bricolage Grotesque", "Replay Body", sans-serif'
    fontSize: "clamp(52px,6.7vw,96px)"
    fontWeight: 550
    lineHeight: 0.98
    letterSpacing: "-.025em"
  headline:
    fontFamily: '"Bricolage Grotesque", "Replay Body", sans-serif'
    fontSize: "clamp(36px,4.2vw,64px)"
    fontWeight: 550
    lineHeight: 1.03
    letterSpacing: "-.025em"
  title:
    fontFamily: '"Bricolage Grotesque", "Replay Body", sans-serif'
    fontSize: "26px"
    fontWeight: 550
    lineHeight: 1.02
    letterSpacing: "-.025em"
  body:
    fontFamily: '"Replay Body", sans-serif'
    fontSize: "16px"
    lineHeight: 1.7
  compact-caption:
    fontFamily: '"Replay Body", sans-serif'
    fontSize: "12px"
  control:
    fontFamily: '"Replay Body", sans-serif'
    fontSize: "14px"
    fontWeight: 600
rounded:
  action: "40px"
  surface: "0"
spacing:
  gutter: "clamp(20px,4vw,72px)"
  section: "clamp(64px,8vw,128px)"
  small: "12px"
  caption: "16px"
  group: "24px"
  block: "32px"
  split: "40px"
  heading: "56px"
  wide-split: "7vw"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.white}"
    typography: "{typography.control}"
    rounded: "{rounded.action}"
    padding: "14px 24px"
  button-primary-hover:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.white}"
  button-light:
    backgroundColor: "{colors.team-home}"
    textColor: "{colors.ink}"
    rounded: "{rounded.action}"
    padding: "14px 24px"
  reel-control:
    backgroundColor: "{colors.team-home}"
    textColor: "{colors.ink}"
    rounded: "{rounded.action}"
    padding: "11px 20px"
  layer-control-selected:
    backgroundColor: "#29533e"
    textColor: "{colors.white}"
    padding: "12px 8px"
  projection-desk:
    backgroundColor: "{colors.field}"
    textColor: "{colors.pitch-line}"
    rounded: "{rounded.surface}"
  programme:
    backgroundColor: "#10291f"
    textColor: "{colors.paper}"
    rounded: "{rounded.surface}"
---

# Design System: footlytics web

## Overview

**Creative North Star: "Football film programme"**

This system is extracted from the final [stylesheet](styles.css), [landing markup](index.html), [About markup](about/index.html), [interactions](site.js), and desktop/mobile review captures. Cream reading paper, field-green stages and chartreuse signals frame real football footage. Expressive medium-weight headings give the work a clear football identity; readable sans paragraphs support technical inspection. Its authority covers the standalone `web/` surface, not the Python pipeline.

Broad media fields, asymmetric working pairs and ruled technical rows carry the visual rhythm. Actual match footage supplies the moving material; synthetic geometry remains explicitly separate. Depth comes from tonal fields and footage rather than ornamental effects. The reusable palette, type and component rules below describe the implemented cascade, not the earlier direction.

The user-authorized real-video redesign supersedes the former pale-steel/navy/rust identity and its square actions/softened desk. The prior document remains in the task baseline. Confirmed constraints survive: real source links, native playback and disclosure, legible mobile order, keyboard focus, reduced motion, and explicit evidence boundaries. This was a code-led build with no approved comp; the fresh finish review records disposition `ship`.

**Key Characteristics:**

- Cream reading fields with dark green film and instrument surfaces.
- Self-hosted Bricolage display above readable sans technical copy.
- Square media, pill actions, thin rules and flat depth.
- Native visitor-controlled film with linked selection and provenance.
- A shared public shell, readable compact captions and ruled startup narrative.

## Colors

The palette joins warm reading paper to green football surfaces, with chartreuse carrying decisive selection and emphasis.

### Primary

- **Field green:** film controls, synthetic instrument and closing band.
- **Forest ink:** main text, display headings and filled primary action.
- **Moss accent:** light-surface link hover, focus and geometry anchors.

### Secondary

- **Chartreuse:** film play action, programme selection, Match State emphasis, roadmap field and the synthetic home-player group. The source retains the `team-home` token name; its role now extends beyond diagram teams.
- **Away cyan:** the opposing player group in the synthetic pitch.

### Neutral

- **Cream paper and white:** reading field and light foreground respectively.
- **Green slate and sage rule:** supporting copy and light-surface separators.
- **Panel sage and chapter sage:** inset technical artifacts and repeated geometry/analytics fields.
- **Pitch line, on-dark muted and on-dark copy:** geometry, supporting copy and captions on green surfaces.
- **Dark rule:** thin divisions across programme, instrument and closing surfaces.

**The Surface Pairing Rule.** Dark green surfaces use their light foregrounds and dark rules; green slate and sage rules belong on reading fields.

**The Signal Rule.** Chartreuse marks an active choice or a deliberate emphasis field; cyan distinguishes the opposing diagram group.

## Typography

**Primary Display Font:** Bricolage Grotesque, user-selected on 09/10/2026. The unmodified variable TTF is self-hosted at `assets/bricolage-grotesque.ttf`, with its adjacent SIL Open Font License. It supports Vietnamese; headings use one font for complete Vietnamese and Latin text.
**Body Font:** Replay Body (Manrope), self-hosted, for paragraphs, navigation, control labels, provenance, addresses and numeric measurements.
**Code Font:** ui-monospace, SFMono-Regular, monospace.

### Hierarchy

- Headings, wordmark and showcase titles use Bricolage Grotesque at weight 550; technology strip uses 500. No synthetic bold. Optical sizing is automatic.
- Hero/About: `clamp(44px,6vw,84px)`, line-height 1.1; tablet `clamp(44px,6vw,62px)`; mobile `clamp(38px,10.5vw,56px)`.
- Chapter headings: `clamp(36px,4.6vw,66px)`, line-height 1.14; mobile `clamp(34px,9vw,48px)`.
- Repeated technical titles: `clamp(24px,2.4vw,32px)`, line-height 1.2. Showcase choice titles are 22px.
- Display tracking is -.015em. The selected scale replaces the prior condensed, heavy Vietnamese role and split Dx Figgle fallback approach.
- Ordinary body text stays 16px on mobile, with line-height 1.8. Captions and controls keep their separate established roles. Paragraph measures stay approximately 44–75ch.
- Both routes preload Bricolage. Dx Figgle and Replay Display are no longer requested by CSS or HTML; legacy asset provenance remains archived.

## Layout

Full-width bands share the fluid gutter and section spacing in the frontmatter. The hero intro, programme, data contract and geometry lab use unequal paired tracks (1.25fr / .7fr), separated by the wide-split spacing. The film band bleeds across the page gutter. Paragraph limits constrain internal text while the working fields remain broad.

The five-stage process begins as columns; tracking and roadmap content use ruled rows with separate heading, explanation and code/status tracks. Tables and code blocks keep local overflow. Heading-to-content separation commonly uses the heading spacing; smaller stacks use the group and block spacing.

At (1000px) and below, paired showcase tracks become (1.15fr / 1fr) with a (36px) gap; the header keeps its contact action beside the wordmark and moves all four navigation links to a second row. Header gaps become (8px / 20px). Technical rows reduce to two tracks. At (640px) and below, intro, programme, data contract and lab stack in reading order, technical rows become single columns, and the process becomes one row per stage. The film precedes its programme. All three synthetic layer controls remain available. The final tablet interval (641–1000px) stages the gallery above a two-column programme: explanation on the left and selection on the right. Hero/About statement pairs use (1fr / .8fr) with (32px) gaps; data contract and geometry lab become full-width stacks. The five-stage process becomes numbered rows with (32px / .7fr / 1fr) tracks. On mobile, each process row keeps number/title together and supporting copy beneath the title. At (1700px) and above, the opening gets more top space while its display holds at (96px).

About keeps the same full-width bands. Its introduction pairs statement and mission at (1.25fr / .7fr), aligned at the bottom with the wide-split gap. Mission and stage rows pair a heading with reading copy at (.7fr / 1.25fr), using one-pixel dividers and (28px) vertical padding. At the tablet breakpoint these gaps become (36px); at the mobile breakpoint both patterns stack with (20px) gaps. The intro padding changes from (64px) to (40px). The real footage still remains broad; the About still uses a mobile (16:8) aspect ratio with its source caption immediately below.

Safe-area insets extend the shared header, section and footer side gutters when necessary; footer bottom spacing also respects the device inset. The mobile gutter is (20px) and section rhythm is (64px). Comparison tables retain a (560px) minimum width inside labelled, keyboard-focusable local scroll regions; ordinary page content stays within the viewport.

**The Band Rule.** Reading limits constrain inner text; the major bands and real media remain full width.

## Elevation & Depth

There is no box-shadow vocabulary. Tonal chapter fields, footage, one-pixel rules and foreground contrast convey depth. Filled action and layer-control color transitions run (160ms) with `cubic-bezier(.16,1,.3,1)`; selection and diagram geometry update immediately. Native film is the primary motion material, started by visitor action with `preload="none"` and no autoplay. The hero pauses when out of view or when the document is hidden.

Reduced motion removes authored transitions and smooth scrolling. Mobile pointer selection may bring the chosen film into view; keyboard selection preserves focus. Native playback remains under visitor control.

**The Flat Field Rule.** Separate working surfaces with tone and rules; retain the footage as the source of visual depth.

## Shapes

Media stages, the projection instrument, technical panels and document bands have square corners. Filled link actions and the hero play control use the action radius; textual actions use an underlined edge. One-pixel dividers organize content. Synthetic pitch geometry uses thin strokes, circular player markers and anchor rings. Selected layers use a two-pixel bottom indicator; selected programme thumbnails use a two-pixel outline with a three-pixel offset.

## Components

### Buttons

Filled links are compact pill actions with text and an inline SVG arrow. The primary variant pairs forest ink and white; the light variant pairs chartreuse and forest ink. Both use the frontmatter padding, minimum height (48px), and moss/white hover. The hero play control uses chartreuse, a pale border, minimum height (46px), and a lighter chartreuse hover. Its inline SVG and text change between play and pause. Text actions use an underlined edge and inline SVG arrow.

Keyboard focus uses a moss outline (3px) with offset (5px) on reading fields; programme-local focus inherits chartreuse. No disabled button variant or input form exists.

### Navigation

The shared header aligns the wordmark, Demo / Công nghệ / Chất lượng / Về startup links and direct email contact. About links point to landing chapters through `/#…`; the About route carries `aria-current="page"` and an underline with (6px) offset. Header and footer links have minimum height (44px); header navigation also has minimum width (44px). Below the tablet breakpoint navigation occupies its own row; on mobile all four links remain available with (14px) labels, (8px / 16px) gaps and (12px) header vertical padding. The separate contact label uses (13px). Below (359px), the four navigation links form a two-column grid.

A ruled local chapter navigation supplies direct technology anchors with minimum height (44px), (8px / 28px) gaps and (13px) labels; mobile uses (8px / 20px) gaps and (12px) labels. Keyboard-visible skip links go to the programme on the landing and main story on About. Footer navigation wraps with (8px / 24px) gaps. Links use real route, anchor, email or source destinations and visible focus.

### Cards / Containers

The recurring working container is the square projection desk: toolbar, pitch, controls and caption on field green. Technical artifacts sit on pale panels. Programme and closing surfaces are broad dark bands. No floating card component or shadowed card stack is present.

### Film Stage and Programme

The hero film fills its band with cropped playback; its caption identifies the ten-second excerpt and supplied annotations. Complete exports use contained native playback and retain their original imagery. Poster thumbnails belong to native selection buttons, with text color and outlined image reinforcing `aria-pressed`.

Selecting one of four films changes video, poster, title, provenance, configuration, limit and direct-file link together. Playback, loading, retry and live-status feedback belong to the selected artifact. The full player exposes browser play, seek and fullscreen controls; its first video and direct links remain usable without JavaScript.

**The Evidence Neighbour Rule.** Keep provenance, configuration and limitations adjacent to the media or measured artifact they qualify.

The public release is `c787ff6d8bbd317f2d14761507dc8a2650556144`, from SoccerTrack v2 / AtomScott and contributors, CC BY 4.0, match 117092. Two films visualize supplied GT and two show unscored predictions; the model examples change both confidence and pitch margin. The legacy radar assumes (105×76m), while the dataset card states (105×68m); this remains unverified. These examples do not prove metric calibration, kinematics or identity accuracy. Historical Brazil–France measurements and Tactical Query roadmap stay separately labeled. Private Alfheim footage is excluded.

### Startup Story and Contact

About mission and development-stage content reuse flat, ruled two-track rows. Titles use condensed display (30px), (28px) on mobile. The progress band uses chapter sage, keeping prototype, validation and roadmap status visibly beside their explanation. The broad still uses the existing ground-truth hero poster; its nearby caption identifies supplied annotations, SoccerTrack v2 / AtomScott and contributors, CC BY 4.0, and links to the complete showcase. The still is source evidence rather than decorative stock imagery.

Contact closes on field green with light reading copy and a chartreuse, underlined address. The address uses the display face at `clamp(24px,3vw,40px)`, weight (650), minimum height (48px) and `overflow-wrap:anywhere`; mobile contact uses `clamp(24px,6.5vw,32px)`. Hover turns the address white. The landing pilot and About share the real user-provided `kickoff@footlytics.space` mailto destination. This is a direct email UI pattern, not a submission form or mailbox-provisioning claim.

**The Shared Route Rule.** Preserve public chapter links and a visible active About state across the landing and startup story.

### Projection Layer Controls

Three native buttons form a labeled group. Minimum height is (48px), with `aria-pressed`, white selected text and a chartreuse bottom indicator. Hover and selected backgrounds use a lighter field tone. Focus uses an inset chartreuse outline. Activation changes geometry, overlays, units and a polite live caption immediately. The visible synthetic label persists in every state.

### Disclosure and Tables

Comparison tables use a labelled region with `tabindex="0"`, contained horizontal scrolling and a visible focus outline. Mobile table copy remains (14px), with (26px) measured values and the local (560px) table width.

Native `details`/`summary` reveals contextual explanation with a ruled boundary and keyboard focus. Measurements use tabular numbers, open rows, hairline rules and larger condensed values. Captions and source notes stay adjacent to the quantities they qualify.

## Do's and Don'ts

### Do:

- **Do** use paired light and dark foregrounds, rules and backgrounds.
- **Do** keep selected film text, thumbnail outline, `aria-pressed` and metadata synchronized.
- **Do** keep complete exports contained and label the cropped hero excerpt.
- **Do** keep GT, prediction, synthetic, historical and roadmap labels adjacent to their artifacts.
- **Do** preserve native playback, local overflow, mobile reading order, keyboard focus and reduced motion.
- **Do** retain all public navigation links on mobile, readable compact captions and visible real email contact.
- **Do** source-label real stills and keep mission, prototype status and roadmap explanations in ruled reading rows.

### Don't:

- **Don't** replace ruled working fields with decorative shadowed card stacks.
- **Don't** use cyan as a generic action color; it distinguishes the opposing synthetic group.
- **Don't** promote compact diagram or provenance annotations into a general reading-text scale.
- **Don't** turn supplied annotations or unscored predictions into claims of validated calibration, kinematics or identity.
- **Don't** present the two prediction clips as a controlled single-parameter experiment.

User-pinned primary-font amendment 09/10/2026: Dx Figgle changes the Latin brand voice within the established film-programme palette and materials. Responsive phone/tablet refinements and the complete Vietnamese display role are scoped in docs/landing/RESPONSIVE.md.
