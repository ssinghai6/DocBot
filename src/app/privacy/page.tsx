// DOCBOT-1707: Privacy Policy + Data retention (draft).
// LEGAL REVIEW REQUIRED: founder-written draft — counsel must review before public launch.
// Retention facts are sourced from code: vector stores are deleted after
// VECTOR_STORE_RETENTION_DAYS (default 30) of inactivity (DOCBOT-1525);
// commerce raw payloads are cleared after COMMERCE_RAW_JSON_RETENTION_DAYS
// (default 90, DOCBOT-1526). Update this page if those defaults change.

import type { Metadata } from "next"
import LegalPage, { COMPANY_NAME, ContactLink, LegalSection } from "@/components/LegalPage"

export const metadata: Metadata = { title: `Privacy Policy — ${COMPANY_NAME}` }

const SUBPROCESSORS: { name: string; purpose: string }[] = [
  { name: "Railway", purpose: "Hosts our backend servers and database." },
  { name: "Vercel", purpose: "Hosts the web app and collects privacy-friendly page analytics." },
  { name: "Groq", purpose: "Runs the AI models that answer your questions." },
  { name: "Google (Gemini)", purpose: "Backup AI model, and financial data extraction from documents." },
  { name: "E2B", purpose: "Isolated sandboxes that run analysis code on your CSV data." },
  { name: "Resend", purpose: "Sends account emails, such as verification links." },
  { name: "Cloudflare Turnstile", purpose: "Bot protection on sign-up." },
  { name: "Stripe", purpose: "Payments and billing. We never see your full card number." },
  { name: "Sentry", purpose: "Error reports, so we can fix bugs." },
  { name: "Hugging Face", purpose: "Only if we turn on hosted embeddings; by default embeddings run on our own servers." },
]

export default function PrivacyPage() {
  return (
    <LegalPage title="Privacy Policy">
      <LegalSection title="Summary">
        <ul>
          <li>We store what you give us so the product works, and nothing more than we need.</li>
          <li><strong>We do not sell your data, and we do not use it to train AI models.</strong></li>
          <li>You can ask us to delete your data at any time.</li>
        </ul>
      </LegalSection>

      <LegalSection title="What we store">
        <ul>
          <li><strong>Account details:</strong> your email, name, and a hashed password (or your GitHub/Google profile ID if you sign in that way), plus your workspace name and the use case you pick at setup.</li>
          <li><strong>Uploaded documents and files:</strong> the text of PDFs you upload, split into chunks with search vectors, and CSV files you upload (stored encrypted).</li>
          <li><strong>Connection credentials:</strong> database connection details and marketplace tokens are encrypted before we store them. They are never written to logs and never sent to AI models.</li>
          <li><strong>Chat history:</strong> your questions, the answers, and results such as charts, SQL and code.</li>
          <li><strong>Usage and security data:</strong> sign-in events, an audit log of actions, AI usage counts for billing, IP address for rate limiting, and error reports.</li>
        </ul>
      </LegalSection>

      <LegalSection title="How we use it">
        <p>
          We use your data to run the Service, keep it secure, bill you, support you, and fix problems. When you ask a
          question, only the text needed to answer it (your question and the relevant excerpts of your documents or
          query results) is sent to the AI provider. Personal details found in results are masked where we detect them.
        </p>
      </LegalSection>

      <LegalSection title="Sub-processors">
        <p>We use these providers to run the Service. Each one only gets the data it needs for its job.</p>
        <ul>
          {SUBPROCESSORS.map((s) => (
            <li key={s.name}><strong>{s.name}</strong> — {s.purpose}</li>
          ))}
        </ul>
      </LegalSection>

      <LegalSection id="data-retention" title="Data retention">
        <ul>
          <li><strong>Uploaded documents and their search vectors:</strong> kept until you delete them or your account. A document session with no activity for 30 days is deleted automatically.</li>
          <li><strong>Anonymous and demo sessions</strong> (used without an account) are short-lived and are deleted by the same 30-day inactivity rule.</li>
          <li><strong>Raw marketplace data</strong> (for example, order payloads from Shopify or Amazon) is cleared after 90 days. The summary figures stay.</li>
          <li><strong>Account details, chat history and connections:</strong> kept while your account is open, and deleted when you delete your account.</li>
          <li><strong>Audit logs and billing records</strong> may be kept longer when security or tax law requires it.</li>
        </ul>
        <p>
          To delete your account and its data, email <ContactLink /> from the address on the account. We will confirm
          once it is done.
        </p>
      </LegalSection>

      <LegalSection title="Security">
        <p>
          Data is encrypted in transit. Connection credentials are encrypted at rest. Database access is read-only, and
          each workspace&apos;s data is kept separate from other workspaces. No system is perfectly secure; if a breach
          affects your data, we will tell you promptly.
        </p>
      </LegalSection>

      <LegalSection title="Your rights">
        <p>
          You can ask to see, correct, export or delete your personal data. Depending on where you live, you may also
          have other rights under laws such as the GDPR or CCPA. Email <ContactLink /> and we will respond within 30 days.
        </p>
      </LegalSection>

      <LegalSection title="Cookies">
        <p>
          We use a small number of cookies to keep you signed in and to protect the Service from abuse. We do not use
          advertising cookies.
        </p>
      </LegalSection>

      <LegalSection title="Changes and contact">
        <p>
          If we change this policy in a material way, we will tell you before the change takes effect. Questions:{" "}
          <ContactLink />.
        </p>
      </LegalSection>
    </LegalPage>
  )
}
