import { useQuery } from '@tanstack/react-query'

import { getCapabilities } from '@/services/apiClient'

export const capabilitiesQueryKey = ['capabilities'] as const

export function useCapabilitiesQuery() {
  return useQuery({
    queryKey: capabilitiesQueryKey,
    queryFn: getCapabilities,
    staleTime: 30_000,
    gcTime: 5 * 60_000,
    refetchInterval: (query) => {
      const cache = query.state.data?.cache
      return cache?.refreshRequested || cache?.refreshing ? 1_000 : 30_000
    },
    refetchIntervalInBackground: true,
    refetchOnMount: 'always',
    refetchOnReconnect: true,
    refetchOnWindowFocus: true,
    structuralSharing: true,
    retry: 1,
  })
}
