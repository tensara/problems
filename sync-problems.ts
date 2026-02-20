import { PrismaClient } from "@prisma/client";
import { readFileSync, readdirSync, existsSync } from "fs";
import path from "path";
import matter from "gray-matter";

const prisma = new PrismaClient();

// Path utility functions
const getProblemsDir = () => path.join(process.cwd(), "problems");
const getProblemPath = (slug: string) =>
  path.join(getProblemsDir(), slug, "problem.md");
const getDefinitionPath = (slug: string) =>
  path.join(getProblemsDir(), slug, "def.py");

// Helper to safely read file contents
const safeReadFile = (path: string): string | null => {
  try {
    return existsSync(path) ? readFileSync(path, "utf8") : null;
  } catch (error) {
    console.warn(`Warning: Could not read file at ${path}`);
    return null;
  }
};

const extractReferenceSolution = (pythonCode: string): string | null => {
  if (!pythonCode) return null;

  const lines = pythonCode.split("\n");
  let inMethod = false;
  let methodLines: string[] = [];
  let methodIndent = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (!inMethod && line.trim().startsWith("def reference_solution(")) {
      inMethod = true;
      methodIndent = line.search(/\S/); // Get the indentation level
      methodLines.push(line);
      continue;
    }

    if (inMethod) {
      const currentIndent = line.search(/\S/);
      const isEmpty = line.trim() === "";

      if (!isEmpty && currentIndent <= methodIndent && currentIndent >= 0) {
        break;
      }

      methodLines.push(line);
    }
  }

  if (methodLines.length === 0) return null;

  const dedentedLines = methodLines.map((line) => {
    if (line.trim() === "") return line;
    return line.slice(methodIndent);
  });

  return dedentedLines.join("\n");
};

const extractParameters = (
  pythonCode: string
): Array<Record<string, unknown>> | null => {
  if (!pythonCode) return null;

  const startMarker = "parameters";
  const idx = pythonCode.indexOf(startMarker);
  if (idx === -1) return null;

  const afterMarker = pythonCode.indexOf("[", idx);
  if (afterMarker === -1) return null;

  let depth = 0;
  let endIdx = -1;
  for (let i = afterMarker; i < pythonCode.length; i++) {
    if (pythonCode[i] === "[") depth++;
    else if (pythonCode[i] === "]") {
      depth--;
      if (depth === 0) {
        endIdx = i;
        break;
      }
    }
  }
  if (endIdx === -1) return null;

  let literal = pythonCode.slice(afterMarker, endIdx + 1);

  literal = literal.replace(/\bTrue\b/g, "true");
  literal = literal.replace(/\bFalse\b/g, "false");
  literal = literal.replace(/\bNone\b/g, "null");
  literal = literal.replace(/'/g, '"');
  literal = literal.replace(/,\s*([}\]])/g, "$1");

  try {
    const parsed = JSON.parse(literal) as Array<Record<string, unknown>>;
    if (!Array.isArray(parsed)) return null;
    return parsed.map((p) => ({
      ...p,
      pointer:
        typeof p.pointer === "boolean" ? String(p.pointer) : p.pointer,
      const: typeof p.const === "boolean" ? String(p.const) : p.const,
    }));
  } catch {
    return null;
  }
};

const extractGetFlops = (pythonCode: string): string | null => {
  if (!pythonCode) return null;

  const lines = pythonCode.split("\n");
  let inMethod = false;
  let methodLines: string[] = [];
  let methodIndent = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (!inMethod && line.trim().startsWith("def get_flops(")) {
      inMethod = true;
      methodIndent = line.search(/\S/); // Get the indentation level
      methodLines.push(line);
      continue;
    }

    if (inMethod) {
      const currentIndent = line.search(/\S/);
      const isEmpty = line.trim() === "";

      if (!isEmpty && currentIndent <= methodIndent && currentIndent >= 0) {
        break;
      }

      methodLines.push(line);
    }
  }

  if (methodLines.length === 0) return null;

  const dedentedLines = methodLines.map((line) => {
    if (line.trim() === "") return line;
    return line.slice(methodIndent);
  });

  return dedentedLines.join("\n");
};

async function main() {
  const problemsDir = getProblemsDir();
  let problemSlugs = readdirSync(problemsDir).filter(
    (slug) => slug !== ".DS_Store" && slug !== "__pycache__"
  );

  const filterSlugs = process.argv.slice(2).filter((arg) => !arg.startsWith("-"));
  if (filterSlugs.length > 0) {
    const valid = new Set(filterSlugs);
    const invalid = filterSlugs.filter((s) => !problemSlugs.includes(s));
    if (invalid.length > 0) {
      console.warn(`Warning: Unknown problem slug(s), skipping: ${invalid.join(", ")}`);
    }
    problemSlugs = problemSlugs.filter((slug) => valid.has(slug));
    console.log(`Syncing ${problemSlugs.length} problem(s): ${problemSlugs.join(", ")}\n`);
  }

  for (const slug of problemSlugs) {
    const problemPath = getProblemPath(slug);

    const fileContents = readFileSync(problemPath, "utf8");
    const { data: frontmatter, content } = matter(fileContents);

    const requiredFields = [
      "slug",
      "title",
      "difficulty",
      "author",
    ];
    const missingFields = requiredFields.filter((field) => !frontmatter[field]);
    if (missingFields.length > 0) {
      throw new Error(
        `Problem ${slug} is missing required frontmatter: ${missingFields.join(
          ", "
        )}`
      );
    }

    const definition = safeReadFile(getDefinitionPath(slug));
    const referenceSolution = definition
      ? extractReferenceSolution(definition)
      : null;
    const getFlops = definition
      ? extractGetFlops(definition)
      : null;

    const defParameters = definition ? extractParameters(definition) : null;
    const parameters = defParameters ?? frontmatter.parameters ?? [];

    if (!parameters || (Array.isArray(parameters) && parameters.length === 0)) {
      console.warn(`  Warning: No parameters found for ${slug} in def.py or problem.md`);
    }

    // Upsert problem in database
    const problem = await prisma.problem.upsert({
      where: { slug },
      update: {
        title: frontmatter.title,
        description: content,
        difficulty: frontmatter.difficulty,
        author: frontmatter.author,
        definition: definition,
        referenceSolution: referenceSolution,
        getFlops: getFlops,
        parameters: parameters,
        tags: frontmatter.tags,
      },
      create: {
        slug,
        title: frontmatter.title,
        description: content,
        difficulty: frontmatter.difficulty,
        author: frontmatter.author,
        definition: definition,
        referenceSolution: referenceSolution,
        getFlops: getFlops,
        parameters: parameters,
        tags: frontmatter.tags,
      },
    });

    const paramSource = defParameters ? "def.py" : (frontmatter.parameters ? "problem.md" : "none");
    console.log(`Synced problem: ${slug}`);
    console.log(`  - Title: ${frontmatter.title ? "✓" : "✗"}`);
    console.log(`  - Difficulty: ${frontmatter.difficulty ? "✓" : "✗"}`);
    console.log(`  - Parameters: ${parameters.length > 0 ? "✓" : "✗"} (source: ${paramSource})`);
    console.log(`  - Definition: ${definition ? "✓" : "✗"}`);
    console.log(`  - Reference Solution: ${referenceSolution ? "✓" : "✗"}`);
    console.log(`  - Get Flops: ${getFlops ? "✓" : "✗"}`);
    console.log(`  - Tags: ${frontmatter.tags ? "✓" : "✗"}`);
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
