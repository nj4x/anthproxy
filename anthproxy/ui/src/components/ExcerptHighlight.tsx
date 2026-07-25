import type { Excerpt } from '../api/types';

interface Props {
  excerpt: Excerpt;
}

export function ExcerptHighlight({ excerpt }: Props) {
  return (
    <span className="text-xs text-gray-500 block mt-0.5 truncate">
      {excerpt.prefix && <span>{excerpt.prefix}</span>}
      <mark className="bg-yellow-200 text-gray-900 rounded-sm px-0.5">{excerpt.match}</mark>
      {excerpt.suffix && <span>{excerpt.suffix}</span>}
    </span>
  );
}
