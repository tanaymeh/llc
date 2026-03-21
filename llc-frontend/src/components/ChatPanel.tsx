import { useEffect, useRef, useState } from 'react';
import type { Components } from 'react-markdown';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { AgentMessageActivity, Message } from '../types';

interface ChatPanelProps {
  messages: Message[];
}

const ACTIVITY_TRANSITION_MS = 170;
const EXTERNAL_LINK_PATTERN = /^(?:[a-z][a-z\d+.-]*:|\/\/)/i;

const markdownComponents = {
  a({ href, children, ...props }) {
    const isExternal = typeof href === 'string' && EXTERNAL_LINK_PATTERN.test(href);
    return (
      <a
        {...props}
        href={href}
        rel={isExternal ? 'noreferrer noopener' : undefined}
        target={isExternal ? '_blank' : undefined}
      >
        {children}
      </a>
    );
  },
} satisfies Components;

const activityPalette = (
  kind: AgentMessageActivity['kind']
): { text: string } => {
  if (kind === 'reasoning') {
    return {
      text: 'var(--activity-reasoning)',
    };
  }
  if (kind === 'tool') {
    return {
      text: 'var(--activity-tool)',
    };
  }
  return {
    text: 'var(--activity-metrics)',
  };
};

interface AgentActivityBadgeProps {
  activity?: AgentMessageActivity;
}

function AgentActivityBadge({ activity }: AgentActivityBadgeProps) {
  const currentTransitionRef = useRef<number | null>(activity?.transitionKey ?? null);
  const transitionTimeoutRef = useRef<number | null>(null);
  const [current, setCurrent] = useState<AgentMessageActivity | null>(activity ?? null);
  const [previous, setPrevious] = useState<AgentMessageActivity | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [dotCount, setDotCount] = useState(1);

  useEffect(
    () => () => {
      if (transitionTimeoutRef.current !== null) {
        window.clearTimeout(transitionTimeoutRef.current);
      }
    },
    []
  );

  useEffect(() => {
    if (!activity) {
      currentTransitionRef.current = null;
      setCurrent(null);
      setPrevious(null);
      setIsTransitioning(false);
      return;
    }

    if (currentTransitionRef.current === null || !current) {
      currentTransitionRef.current = activity.transitionKey;
      setCurrent(activity);
      setPrevious(null);
      setIsTransitioning(false);
      return;
    }

    if (activity.transitionKey === currentTransitionRef.current) {
      setCurrent(activity);
      return;
    }

    if (transitionTimeoutRef.current !== null) {
      window.clearTimeout(transitionTimeoutRef.current);
      transitionTimeoutRef.current = null;
    }

    setPrevious(current);
    setCurrent(activity);
    setIsTransitioning(false);
    currentTransitionRef.current = activity.transitionKey;
    window.requestAnimationFrame(() => {
      setIsTransitioning(true);
    });
    transitionTimeoutRef.current = window.setTimeout(() => {
      setPrevious(null);
      setIsTransitioning(false);
      transitionTimeoutRef.current = null;
    }, ACTIVITY_TRANSITION_MS + 40);
  }, [activity, current]);

  useEffect(() => {
    if (current?.kind !== 'reasoning') {
      return undefined;
    }
    setDotCount(1);
    const interval = window.setInterval(() => {
      setDotCount((prev) => (prev >= 3 ? 1 : prev + 1));
    }, 320);
    return () => {
      window.clearInterval(interval);
    };
  }, [current?.kind, current?.transitionKey]);

  if (!current && !previous) {
    return null;
  }

  const renderLabel = (entry: AgentMessageActivity, isLiveEntry: boolean) => {
    if (entry.kind !== 'reasoning') {
      return entry.label;
    }
    const dots = '.'.repeat(isLiveEntry ? dotCount : 3);
    return `Reasoning${dots}`;
  };

  const renderBadge = (
    entry: AgentMessageActivity,
    variant: 'current' | 'previous',
    label: string
  ) => {
    const palette = activityPalette(entry.kind);
    const baseClass = 'absolute left-0 top-0 inline-flex h-5 items-center whitespace-nowrap text-[11px] tracking-[0.12em] transition-all ease-out';
    const transitionClass = variant === 'previous'
      ? (isTransitioning ? '-translate-y-5 opacity-0' : 'translate-y-0 opacity-100')
      : (
        previous
          ? (isTransitioning ? 'translate-y-0 opacity-100' : 'translate-y-5 opacity-0')
          : 'translate-y-0 opacity-100'
      );
    return (
      <span
        className={`${baseClass} ${transitionClass}`}
        style={{
          transitionDuration: `${ACTIVITY_TRANSITION_MS}ms`,
        }}
      >
        <span className="theme-text-faint mr-1">|</span>
        <span style={{ color: palette.text }}>{label}</span>
      </span>
    );
  };

  const previousLabel = previous ? renderLabel(previous, false) : '';
  const currentLabel = current ? renderLabel(current, true) : '';
  const sizingLabel = currentLabel.length >= previousLabel.length ? currentLabel : previousLabel;

  return (
    <div className="relative inline-grid h-5 shrink-0 overflow-x-visible overflow-y-hidden">
      <span className="invisible inline-flex h-5 items-center whitespace-nowrap text-[11px] tracking-[0.12em]">
        <span className="mr-1">|</span>
        <span>{sizingLabel}</span>
      </span>
      {previous && renderBadge(previous, 'previous', previousLabel)}
      {current && renderBadge(current, 'current', currentLabel)}
    </div>
  );
}

export function ChatPanel({ messages }: ChatPanelProps) {
  const getMessageLabelColor = (message: Message) => {
    if (message.role === 'USER') {
      return 'theme-text-primary';
    }
    if (message.role === 'AGENT') {
      return 'text-[var(--accent-success)]';
    }

    switch (message.tone) {
      case 'enabled':
        return 'text-[var(--accent-success)]';
      case 'disabled':
        return 'text-[var(--accent-warning)]';
      case 'selection':
        return 'text-[var(--accent-selection)]';
      case 'error':
        return 'text-[var(--accent-error)]';
      default:
        return 'theme-text-secondary';
    }
  };

  const getMessageContentColor = (message: Message) => {
    if (message.role !== 'SYSTEM') {
      return 'theme-text-secondary';
    }

    switch (message.tone) {
      case 'enabled':
        return 'text-[var(--accent-success)]';
      case 'disabled':
        return 'text-[var(--accent-warning)]';
      case 'selection':
        return 'text-[var(--accent-selection)]';
      case 'error':
        return 'text-[var(--accent-error)]';
      default:
        return 'theme-text-secondary';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="theme-text-muted text-xs font-mono">{message.timestamp}</span>
              <span className={`text-xs tracking-wider ${getMessageLabelColor(message)}`}>
                [{message.role}]
              </span>
              {message.role === 'AGENT' && (
                <AgentActivityBadge activity={message.activity} />
              )}
            </div>
            {message.role === 'AGENT' ? (
              <div className={`${getMessageContentColor(message)} mission-markdown text-sm font-mono leading-relaxed`}>
                <ReactMarkdown
                  components={markdownComponents}
                  remarkPlugins={[remarkGfm]}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            ) : (
              <p className={`${getMessageContentColor(message)} text-sm font-mono leading-relaxed whitespace-pre-wrap break-words`}>
                {message.content}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
