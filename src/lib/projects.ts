// Instead of direct DB connection, we'll use static data for now
export const projects = [
    {
        id: 1,
        title: "Enterprise MLOps Pipeline",
        description: "Production-grade MLOps system implementing continuous training, model deployment, monitoring, and automated model lifecycle management",
        slug: "mlops-pipeline",
        hero_image: "/post_img.webp",
        technologies: ["Python", "MLflow", "Kubernetes"],
        github_url: "https://github.com/yourusername/mlops-pipeline",
        live_url: null
    },
    {
        id: 2,
        title: "Knowledge Graph System",
        description: "Enterprise-grade knowledge graph platform implementing entity extraction, relationship mining, and semantic reasoning",
        slug: "knowledge-graph",
        hero_image: "/post_img.webp",
        technologies: ["Python", "Neo4j", "NLP"],
        github_url: "https://github.com/yourusername/knowledge-graph",
        live_url: null
    },
    {
        id: 3,
        title: "Advanced Anomaly Detection System",
        description: "Enterprise-grade anomaly detection system using deep learning, statistical methods, and real-time processing for multivariate time series data",
        slug: "anomaly-detection",
        hero_image: "/post_img.webp",
        technologies: ["Python", "TensorFlow", "Kafka"],
        github_url: "https://github.com/yourusername/anomaly-detection",
        live_url: null
    }
];

export const blogPosts = [
    {
        id: 1,
        title: "The Intelligent Investor: Benjamin Graham's Timeless Principles",
        description: "A deep dive into Benjamin Graham's foundational work on value investing",
        slug: "intelligent-investor",
        hero_image: "/post_img.webp",
        published_at: "2025-02-10T00:00:00Z",
        tags: ["Finance", "Investing"]
    },
    {
        id: 2,
        title: "Howard Marks' Investment Philosophy",
        description: "Essential wisdom from Howard Marks on market cycles and risk management",
        slug: "howard-marks-wisdom",
        hero_image: "/post_img.webp",
        published_at: "2025-02-09T00:00:00Z",
        tags: ["Finance", "Investing", "Risk Management"]
    },
    {
        id: 3,
        title: "Modern MLOps Architecture",
        description: "Building scalable machine learning operations for enterprise",
        slug: "modern-mlops",
        hero_image: "/post_img.webp",
        published_at: "2025-02-08T00:00:00Z",
        tags: ["MLOps", "AI", "Engineering"]
    }
];

export async function getProjects() {
    return projects;
}

export async function getBlogPosts() {
    return blogPosts;
} 