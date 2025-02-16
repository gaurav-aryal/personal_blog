import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  site: 'https://gauravaryal.com',
  integrations: [mdx(), sitemap(), tailwind()],
  markdown: {
    shikiConfig: {
      theme: 'one-dark-pro',
      wrap: true
    },
    remarkPlugins: [],
    rehypePlugins: []
  },
  experimental: {
    contentCollections: true
  }
});