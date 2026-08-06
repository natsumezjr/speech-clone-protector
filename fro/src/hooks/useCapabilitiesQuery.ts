import { useQuery } from '@tanstack/react-query'

import { getCapabilities } from '@/services/apiClient'

export const capabilitiesQueryKey = ['capabilities'] as const

export function useCapabilitiesQuery() {
  return useQuery({
    queryKey: capabilitiesQueryKey,
    queryFn: getCapabilities,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: Number.POSITIVE_INFINITY,
    refetchOnMount: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
    retry: 1,
  })
}
