import { ReactNode } from 'react';
import { clsx } from 'clsx';

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;  // Optional action element (e.g., button)
}

export function Card({ title, children, className, action }: CardProps) {
  return (
    <div className={clsx('bg-white rounded-lg shadow-md', className)}>
      {title && (
        <div className="px-3 py-1.5 border-b border-gray-200 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-2.5">{children}</div>
    </div>
  );
}
