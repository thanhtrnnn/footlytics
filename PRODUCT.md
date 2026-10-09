# footlytics

## Platform
Web landing for a Python football video-analysis prototype.

## Users and job
Analysts and clubs evaluate a video-to-pitch-data pipeline; technical collaborators inspect its mechanisms, artifacts and limits. Analyst/pilot audience is a proposal from existing product docs, not verified customer traction.

## Product truth
Video → detection/calibration → tracking in metres → Match State → analytics/quality/radar. Match State uses 17 columns and a common data contract. Moving-camera support uses KLT/SIFT re-anchoring. Identity and referee separation remain limited.
Tactical Query is a roadmap: possession inference, validated sequences, multi-match index/query are not a complete shipped product. No realtime/90-minute/OCR-complete claims.

## Evidence
docs/landing/CLAIMS.md and TECHNICAL.md separate local code from canonical September 23 documentation at thanhtrnnn/footlytics. Historical F2 presence percentages are not accuracy, and have not been rerun for this landing.

## Request and constraints
The user requests a one-shot build of both websites, maximizing selected technical angles. The latest request replaces Footlytics' visual world with a real-video showcase inspired by VGF and Studio DUY, and authorizes publishing it to footlytics.space. Preserve provenance and prototype/roadmap labels. Contact confirmed by the user: kickoff@footlytics.space. Founders, customers and pricing remain unconfirmed. No fake forms or fabricated demos.

Public demo release c787ff6: two supplied-ground-truth visualizations and two unscored predictions from SoccerTrack v2 / AtomScott and contributors, CC BY 4.0, match 117092. The model clips vary both confidence and pitch margin. Legacy radar assumes 105×76m, differing from the current dataset card's 105×68m; do not claim verified calibration/kinematics/identity from these clips. Historical Brazil–France measurements remain separate. Private Alfheim footage is excluded. Four uncropped exports are preserved; a labelled ten-second top-video crop provides the hero reel.

## Stack
Python pipeline stays unchanged. The new standalone landing uses static HTML/CSS/JavaScript for Cloudflare Pages. This implementation choice is delegated by the one-shot build request; it is not a migration of the pipeline.

## Brand
No confirmed web identity in the repository. Create a distinct football-analysis identity; do not copy drakefamtax. Illustrative tactical diagrams must be labeled conceptual/synthetic wherever they resemble pipeline output.

## Success
Visitors can explain the mechanism, find measurements and limitations, and distinguish prototype from Tactical Query roadmap. Responsive, accessible site with real anchors/source links and deployable static files.


Public presentation update09/10/2026: User pinned primary Dx Figgle and confirmed WebFontlicense for footlytics.space. Selfhosted unmodified OTF400 with complete Vietnamese ReplayDisplay700 role for unsupported glyphs. Responsive adapts existing film-programme; no pipeline changes.

Typography selection 09/10/2026: User replaced Dx Figgle with Bricolage Grotesque after reviewing a font comparison. All public display headings/brand use Bricolage 550, technology labels 500; body remains Manrope. This selection supersedes the prior Dx Figgle/Replay Display pin.
