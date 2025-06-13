import { PrismaClient } from "@prisma/client";
import { readFileSync, readdirSync } from "fs";
import path from "path";
import { runBaselineBenchmark } from "./baselineBenchmark";

const prisma = new PrismaClient();

const baselineFiles = {
  "tinygrad": "tinygrad",
  "torch_compile": "torch",
  "torch_vanilla": "torch",
}

const args = process.argv.slice(2);
const targetBaselines = args.length > 0 ? args : Object.keys(baselineFiles);
const targetProblems = args.find(arg => arg.startsWith("--problems="))?.split("=")[1]?.split(",");

const getProblemsDir = () => path.join(process.cwd(), "problems");
const getSolutionPath = (slug: string, baseline: string) =>
  path.join(getProblemsDir(), slug, `${baseline}.py`);
const getDefinitionPath = (slug: string) =>
  path.join(getProblemsDir(), slug, "def.py");

async function main() {
  const problemsDir = getProblemsDir();
  const problemSlugs = readdirSync(problemsDir)
    .filter((slug) => slug !== ".DS_Store" && slug !== "__pycache__")
    .filter((slug) => !targetProblems || targetProblems.includes(slug));

  console.log(`Running benchmarks for baselines: ${targetBaselines.join(", ")}`);
  if (targetProblems) {
    console.log(`Targeting specific problems: ${targetProblems.join(", ")}`);
  }

  for (const slug of problemSlugs) {    
    const definitionPath = getDefinitionPath(slug);
    let definition;
    try {
      definition = readFileSync(definitionPath, "utf8");
    } catch (error) {
      console.warn(`Warning: Could not read definition for ${slug}`);
      continue;
    }

    for (const baseline of targetBaselines) {
      if (!baselineFiles[baseline]) {
        console.warn(`Warning: Unknown baseline type "${baseline}". Skipping...`);
        continue;
      }

      const solutionPath = getSolutionPath(slug, baselineFiles[baseline]);
      let solution;
      
      try {
        solution = readFileSync(solutionPath, "utf8");
      } catch (error) {
        console.warn(`Warning: No ${baseline} baseline found for ${slug}`);
        continue;
      }
      
      const result = await runBaselineBenchmark({
        code: solution,
        problemSlug: slug,
        problemDefinition: definition,
        language: "python",
        modalEndpoint: process.env.BASELINE_ENDPOINT as string,
        baseline: baseline,
        gpuType: "T4",
        dtype: "float32"
      });

      if (result.success && result.results) {        
        const existingProblem = await prisma.problem.findUnique({
          where: { slug }
        });
        
        const existingBenchmarks = (existingProblem?.baselineBenchmarks as Record<string, any>) || {};
        
        await prisma.problem.update({
          where: {
            slug: slug,
          },
          data: {
            baselineBenchmarks: {
              ...existingBenchmarks,
              [baseline]: {
                results: result.results,
                avg_gflops: result.avg_gflops,
                avg_runtime_ms: result.avg_runtime_ms,
                total_tests: result.total_tests
              }
            }
          }
        });

        console.log(`✓ ${baseline} benchmark completed and saved`);
        console.log(`  - Tests: ${result.results.length}`);
        const avgGflops = result.results.reduce((sum, r) => sum + r.gflops, 0) / result.results.length;
        console.log(`  - Avg GFLOPS: ${avgGflops.toFixed(2)}`);
      } else {
        console.error(`✗ ${baseline} benchmark failed:`, result.error);
        if (result.error) {
          console.error(`  Details: ${result.error}`);
        }
      }
    }
  }
}

main()
  .catch((e) => {
    console.error("❌ Sync failed:", e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  }); 