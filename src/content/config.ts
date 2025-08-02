import { z, defineCollection } from "astro:content";

const blogSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.string(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    tags: z.array(z.string()).optional(),
});

const projectsSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.string(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    tags: z.array(z.string()).optional(),
});

const neuralnexusSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
});

const storeSchema = z.object({
    title: z.string(),
    description: z.string(),
    details: z.boolean().optional(),
    custom_link_label: z.string(),
    custom_link: z.string().optional(),
    updatedDate: z.coerce.date(),
    pricing: z.string().optional(),
    oldPricing:  z.string().optional(),
    badge: z.string().optional(),
    checkoutUrl: z.string().optional(),
    heroImage: z.string().optional(),
});

const financeSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.string(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
    tags: z.array(z.string()).optional(),
});

const leetcodeSchema = z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    order: z.number().optional(),
});

const systemDesignSchema = z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    order: z.number(),
});

const blind75Schema = z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    order: z.number().optional(),
});


export type BlogSchema = z.infer<typeof blogSchema>;
export type StoreSchema = z.infer<typeof storeSchema>;
export type ProjectsSchema = z.infer<typeof projectsSchema>;
export type NeuralNexusSchema = z.infer<typeof neuralnexusSchema>;
export type FinanceSchema = z.infer<typeof financeSchema>;
export type LeetcodeSchema = z.infer<typeof leetcodeSchema>;
export type SystemDesignSchema = z.infer<typeof systemDesignSchema>;
export type Blind75Schema = z.infer<typeof blind75Schema>;

const blogCollection = defineCollection({ schema: blogSchema });
const storeCollection = defineCollection({ schema: storeSchema });
const projectCollection = defineCollection({ schema: projectsSchema });
const neuralnexusCollection = defineCollection({ schema: neuralnexusSchema });
const financeCollection = defineCollection({ schema: financeSchema });
const leetcodeCollection = defineCollection({ schema: leetcodeSchema });
const systemDesignCollection = defineCollection({ schema: systemDesignSchema });
const blind75Collection = defineCollection({ schema: blind75Schema });

const projectsCollection = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        description: z.string(),
        pubDate: z.string(),
        heroImage: z.string()
    })
});

export const collections = {
    'blog': blogCollection,
    'store': storeCollection,
    'projects': projectsCollection,
    'neuralnexus': neuralnexusCollection,
    'finance': financeCollection,
    'leetcode': leetcodeCollection,
    'system-design': systemDesignCollection,
    'blind75': blind75Collection,
}