interface LogoMarkProps {
  className?: string
  variant?: "full" | "compact"
}

const PLATFORM_TEAL = "#2DD4BF"
const SPARK_AMBER = "#F5C66B"
const OUTLINE_FULL = "M12.4 2.6H39.8L51.6 15.4V49.4A19.6 5 0 0 1 12.4 49.4V2.6Z"
const OUTLINE_COMPACT = "M12.4 2.6H51.6V49.4A19.6 5 0 0 1 12.4 49.4V2.6Z"
const FOLD_FLAP = "M39.8 2.6V10.4A2.2 2.2 0 0 0 42 12.6H51.6"
const CYLINDER_RIM = "M51.6 33.4A19.6 7.2 0 0 1 12.4 33.4"
const SPARK =
  "M27.1 12C27.78 15.48 29.49 17.28 32.8 18C29.49 18.72 27.78 20.52 27.1 24C26.42 20.52 24.71 18.72 21.4 18C24.71 17.28 26.42 15.48 27.1 12Z"

export default function LogoMark({ className = "w-4 h-4", variant = "compact" }: LogoMarkProps) {
  const full = variant === "full"
  const stroke = full ? 5.2 : 5.4
  return (
    <svg viewBox="9.8 0 44.4 57" fill="none" className={className} aria-hidden="true">
      <path
        d={full ? OUTLINE_FULL : OUTLINE_COMPACT}
        stroke={PLATFORM_TEAL}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {full && (
        <path
          d={FOLD_FLAP}
          stroke={PLATFORM_TEAL}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
      <path d={CYLINDER_RIM} stroke={PLATFORM_TEAL} strokeWidth={stroke} strokeLinecap="round" />
      <path d={SPARK} fill={SPARK_AMBER} />
    </svg>
  )
}
