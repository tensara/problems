export type TestResult = {
  test_id: number;
  name: string;
  gflops: number;
  runtime_ms: number;
};

export async function runBaselineBenchmark(params: {
  code: string;
  problemSlug: string;
  problemDefinition: any;
  gpuType?: string;
  language: string;
  modalEndpoint: string;
  baseline: string;
  dtype?: string;
}): Promise<{
  success: boolean;
  results?: TestResult[];
  error?: string;
  avg_gflops?: number;
  avg_runtime_ms?: number;
  total_tests?: number;
}> {
  try {
    const response = await fetch(
      `${params.modalEndpoint}/${params.baseline}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          solution_code: params.code,
          problem: params.problemSlug,
          problem_def: params.problemDefinition,
          gpu: params.gpuType ?? "t4",
          dtype: params.dtype ?? "float32",
          language: params.language,
        }),
      }
    );

    if (!response.ok) {
      return {
        success: false,
        error: `API returned ${response.status}: ${await response.text()}`,
      };
    }

    const reader = response.body?.getReader();
    if (!reader) {
      return {
        success: false,
        error: "No response body from API",
      };
    }

    let partialMessage = "";
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = new TextDecoder().decode(value);
        partialMessage += text;

        const messages = partialMessage.split("\n\n");
        partialMessage = messages.pop() ?? "";

        for (const message of messages) {
          if (!message?.startsWith("data: ")) continue;

          const data = JSON.parse(message.slice(6).trim());
          if (data.status === "BENCHMARKED") {
            return {
              success: true,
              results: data.test_results,
              avg_gflops: data.avg_gflops,
              avg_runtime_ms: data.avg_runtime_ms,
              total_tests: data.total_tests,
            };
          } else if (data.error) {
            return {
              success: false,
              error: data.error,
            };
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    return {
      success: true,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}
