// DOCBOT-1707: Terms of Service (draft).
// LEGAL REVIEW REQUIRED: founder-written draft — counsel must review before public launch.

import type { Metadata } from "next"
import Link from "next/link"
import LegalPage, { COMPANY_NAME, ContactLink, LegalSection } from "@/components/LegalPage"

export const metadata: Metadata = { title: `Terms of Service — ${COMPANY_NAME}` }

export default function TermsPage() {
  return (
    <LegalPage title="Terms of Service">
      <LegalSection title="1. Agreement">
        <p>
          These terms govern your use of {COMPANY_NAME} (the &ldquo;Service&rdquo;). By creating an account or using the
          Service you agree to them. If you use the Service for a company, you confirm you can accept these terms on
          its behalf.
        </p>
      </LegalSection>

      <LegalSection title="2. The Service">
        <p>
          {COMPANY_NAME} is an early-stage product that analyses documents and data you provide, including checking
          figures in a report against your books. We change and improve it often, and some features may be marked beta.
        </p>
      </LegalSection>

      <LegalSection title="3. Your account">
        <ul>
          <li>Give accurate sign-up details and keep your password safe.</li>
          <li>You are responsible for activity under your account. Tell us promptly if you think it has been misused.</li>
          <li>You must be at least 18 years old to use the Service.</li>
        </ul>
      </LegalSection>

      <LegalSection title="4. Your data">
        <p>
          You keep all rights to the documents, data and database connections you bring (&ldquo;Customer
          Data&rdquo;). You give us permission to store and process Customer Data only to run the Service for you.
          <strong> We do not use Customer Data to train AI models.</strong> How we handle it is described in the{" "}
          <Link href="/privacy" className="text-[var(--color-cyan-500)] hover:underline">Privacy Policy</Link>.
        </p>
        <p>You confirm you have the right to upload or connect the data you use with the Service.</p>
      </LegalSection>

      <LegalSection title="5. Acceptable use">
        <p>You agree not to:</p>
        <ul>
          <li>break the law, or upload data you have no right to share;</li>
          <li>try to access other customers&apos; data, or probe, scan or attack the Service;</li>
          <li>overload the Service, get around usage limits, or resell it without our permission;</li>
          <li>use the Service to build a competing product.</li>
        </ul>
        <p>We may suspend accounts that break these rules.</p>
      </LegalSection>

      <LegalSection title="6. AI output">
        <p>
          The Service uses AI models, and their answers can be wrong. Check results before you rely on them for
          financial reporting, filings, or other decisions. The Service does not give accounting, legal, tax or
          investment advice.
        </p>
      </LegalSection>

      <LegalSection title="7. Fees">
        <p>
          Paid plans are billed through Stripe at the price shown when you subscribe. Unless the plan says otherwise,
          fees are non-refundable and you can cancel at any time, effective at the end of the current billing period.
        </p>
      </LegalSection>

      <LegalSection title="8. Ending your use">
        <p>
          You can stop using the Service and ask us to delete your account at any time. We may suspend or end your
          access if you break these terms or if we shut down the Service, and we will try to give reasonable notice
          so you can export your work.
        </p>
      </LegalSection>

      <LegalSection title="9. Disclaimers and liability">
        <p>
          The Service is provided &ldquo;as is&rdquo; without warranties of any kind. To the extent the law allows,{" "}
          {COMPANY_NAME} is not liable for indirect or consequential losses, and our total liability is limited to the
          amount you paid us in the 12 months before the claim.
        </p>
      </LegalSection>

      <LegalSection title="10. Changes">
        <p>
          We may update these terms. If a change is material, we will tell you by email or in the product before it
          takes effect. If you keep using the Service after that, you accept the updated terms.
        </p>
      </LegalSection>

      <LegalSection title="11. Contact">
        <p>Questions about these terms: <ContactLink />.</p>
      </LegalSection>
    </LegalPage>
  )
}
