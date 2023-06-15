import { z, defineCollection } from "astro:content";

const blogSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
});

const projectsSchema = z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
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
    pubDate: z.coerce.date(),
    updatedDate: z.string().optional(),
    heroImage: z.string().optional(),
});


export type BlogSchema = z.infer<typeof blogSchema>;
export type StoreSchema = z.infer<typeof storeSchema>;
export type ProjectsSchema = z.infer<typeof projectsSchema>;
export type NeuralNexusSchema = z.infer<typeof neuralnexusSchema>;
export type FinanceSchema = z.infer<typeof financeSchema>;

const blogCollection = defineCollection({ schema: blogSchema });
const storeCollection = defineCollection({ schema: storeSchema });
const projectCollection = defineCollection({ schema: projectsSchema });
const neuralnexusCollection = defineCollection({ schema: neuralnexusSchema });
const financeCollection = defineCollection({ schema: financeSchema });

export const collections = {
    'blog': blogCollection,
    'store': storeCollection,
    'projects': projectCollection,
    'neuralnexus': neuralnexusCollection,
    'finance': financeCollection
}