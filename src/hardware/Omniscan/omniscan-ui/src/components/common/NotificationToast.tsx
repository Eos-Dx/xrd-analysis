import { useEffect } from 'react';
import { useStore } from '@/store/useStore';
import { clsx } from 'clsx';

export function NotificationToast() {
  const { notifications, removeNotification } = useStore();

  useEffect(() => {
    notifications.forEach((notification) => {
      const timer = setTimeout(() => {
        removeNotification(notification.id);
      }, 5000);

      return () => clearTimeout(timer);
    });
  }, [notifications, removeNotification]);

  if (notifications.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => {
        const styles = {
          info: 'bg-blue-50 border-blue-500 text-blue-900',
          success: 'bg-green-50 border-green-500 text-green-900',
          warning: 'bg-yellow-50 border-yellow-500 text-yellow-900',
          error: 'bg-red-50 border-red-500 text-red-900',
        };

        return (
          <div
            key={notification.id}
            className={clsx(
              'min-w-80 rounded-lg border-l-4 p-4 shadow-lg',
              styles[notification.type]
            )}
          >
            <div className="flex items-start justify-between">
              <p className="font-medium">{notification.message}</p>
              <button
                onClick={() => removeNotification(notification.id)}
                className="ml-4 text-gray-500 hover:text-gray-700"
              >
                ×
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
