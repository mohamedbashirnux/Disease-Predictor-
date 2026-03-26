# Deploy to Vercel

## Steps:

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Vercel:**
   - Go to https://vercel.com
   - Sign in with GitHub
   - Click "New Project"
   - Import your repository
   - Click "Deploy"
   - Done! Vercel will give you a URL

## Important Notes:

- All your model files in `models/` folder will be deployed
- The `data/` folder will also be included
- First deployment might take 2-3 minutes
- Vercel free tier is perfect for this project

## After Deployment:

Your app will be live at: `https://your-project-name.vercel.app`

## Troubleshooting:

If deployment fails:
- Check the build logs in Vercel dashboard
- Make sure all files are committed to GitHub
- Verify `requirements.txt` has all dependencies
