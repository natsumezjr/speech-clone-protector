import { useQuery } from '@tanstack/react-query'

import { getCapabilities } from '@/services/apiClient'

export const capabilitiesQueryKey = ['capabilities'] as const

export function useCapabilitiesQuery() {
  return useQuery({
    queryKey: capabilitiesQueryKey,
    queryFn: getCapabilities,
    staleTime: 30_000,
    gcTime: 5 * 60_000,
    refetchOnMount: 'always',
    refetchOnReconnect: true,
    refetchOnWindowFocus: true,
    retry: 1,
  })
}
