// DOCBOT-2337: the nonce CSP has no 'unsafe-eval' in prod. Zod v4 otherwise
// probes `new Function("")` on first parse, which the browser reports as a
// CSP violation even though Zod catches it. jitless skips the probe.
import { z } from "zod";

z.config({ jitless: true });
