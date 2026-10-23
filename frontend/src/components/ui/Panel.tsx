import type { ReactNode } from "react";
import { ArrowDownRight, LoaderCircle } from "lucide-react";

export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <section className={`work-panel ${className}`}>{children}</section>;
}

export function PanelHeading({
  icon,
  title,
  trailing,
}: {
  icon: ReactNode;
  title: string;
  trailing?: ReactNode;
}) {
  return (
    <div className="panel-heading">
      <div className="panel-title">
        {icon}
        <h2>{title}</h2>
      </div>
      {trailing}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="empty-state">
      <div className="empty-symbol">
        <ArrowDownRight size={19} />
      </div>
      <strong>{title}</strong>
      <span>{hint}</span>
    </div>
  );
}

export function LoadingState() {
  return (
    <div className="loading-state">
      <LoaderCircle size={23} className="spin" />
      <strong>Running experiment</strong>
      <span>Processing with the selected classical NLP method…</span>
    </div>
  );
}

export function Metric({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string | number;
  accent?: boolean;
}) {
  return (
    <div className={`metric-card ${accent ? "accent" : ""}`}>
      <span>{label}</span>
      <strong>
        {typeof value === "number" ? value.toLocaleString() : value}
      </strong>
    </div>
  );
}

export function SectionLabel({
  children,
  trailing,
}: {
  children: ReactNode;
  trailing?: ReactNode;
}) {
  return (
    <div className="section-label">
      {children}
      {trailing && <span>{trailing}</span>}
    </div>
  );
}

export function NoteBox({ children }: { children: ReactNode }) {
  return (
    <div className="note-box">
      {children}
    </div>
  );
}
