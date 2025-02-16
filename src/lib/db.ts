import { Pool } from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const pool = new Pool({
    user: process.env.POSTGRES_USER,
    password: process.env.POSTGRES_PASSWORD,
    host: process.env.POSTGRES_HOST,
    port: parseInt(process.env.POSTGRES_PORT || '5432'),
    database: process.env.POSTGRES_DB,
});

export async function query(text: string, params?: any[]) {
    return pool.query(text, params);
}

export async function getProjects() {
    const result = await query(`
        SELECT * FROM projects 
        ORDER BY created_at DESC
    `);
    return result.rows;
}

export async function getBlogPosts() {
    const result = await query(`
        SELECT * FROM blog_posts 
        ORDER BY published_at DESC
    `);
    return result.rows;
} 