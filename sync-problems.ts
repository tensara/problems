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
  const problemSlugs = readdirSync(problemsDir).filter(
    (slug) => slug !== ".DS_Store" && slug !== "__pycache__"
  );

  for (const slug of problemSlugs) {
    const problemPath = getProblemPath(slug);

    const fileContents = readFileSync(problemPath, "utf8");
    const { data: frontmatter, content } = matter(fileContents);

    const requiredFields = [
      "slug",
      "title",
      "difficulty",
      "author",
      "parameters",
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
        parameters: frontmatter.parameters,
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
        parameters: frontmatter.parameters,
        tags: frontmatter.tags,
      },
    });

    console.log(`Synced problem: ${slug}`);
    console.log(`  - Title: ${frontmatter.title ? "✓" : "✗"}`);
    console.log(`  - Difficulty: ${frontmatter.difficulty ? "✓" : "✗"}`);
    console.log(`  - Parameters: ${frontmatter.parameters ? "✓" : "✗"}`);
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
