import type { Metadata } from "next"
import ApisPortal from "@/components/apis/ApisPortal"

export const metadata: Metadata = {
  title: "DocBot API | Developer portal",
  description:
    "DocBot API reference and playground. Get a free test key, spin up a sandbox and ask your first hybrid docs + database question.",
}

export default function ApisPage() {
  return <ApisPortal />
}
