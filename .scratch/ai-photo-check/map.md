# AI Photo Check MVP

## Destination

A working, free, English-language responsive web app for the general public to assess one uploaded photo at a time for AI generation or post-capture AI editing, without claiming to certify authenticity.

## Notes

- Domain: public-facing image assessment.
- Carry this effort through delivery of the MVP once the decisions needed to build it are clear.
- Consult `CONTEXT.md` for domain language; consult domain-modeling when resolving new terms.
- Default to transient upload processing. Any private-history retention is opt-in; training or research requires separate consent.
- Keep the product informational, calibrated, and unsuitable as the sole basis for high-stakes decisions.

## Decisions so far

<!-- Closed decision tickets will be indexed here; each answer lives in its ticket. -->

## Not yet specified

- Whether current detection methods can credibly assess both fully AI-generated images and the broad range of post-capture AI enhancement and retouching, and what limitations they impose.
- The exact evidence and explanation shown for detected signals, no reliable signals, and inconclusive outcomes.
- Operational details of upload processing, optional private history, consent, logs, deletion, and any external processing.
- Which image formats, sizes, and quality levels are supportable, plus behavior for unsupported, corrupted, or inconclusive inputs.
- The assessment summary that can safely be exported or shared.
- A suitable inference and storage approach for a free public MVP.
- Release thresholds, evaluation methodology, and safeguards for public use.

## Out of scope

- Native mobile apps and non-English localization for this MVP.
- URL-based, batch, and video checks; the MVP checks one device-uploaded image at a time.
- User accounts, paid plans, and monetization.
- High-stakes forensic, legal, employment, safety, or moderation decisions.
- Certifying that an image is authentic, or identifying whether it depicts a real event.
- Automatic AI processing performed by a camera during capture.
