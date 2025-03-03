# Sky Theme Installation Instructions

Follow these steps to add the sky-themed background to your GitHub.io website:

## 1. Create the required directories

If they don't already exist, create these directories in your repository:
```
assets/
assets/css/
assets/images/
```

## 2. Add the CSS files

Place the `sky-theme.css` file in the `assets/css/` directory.

## 3. Add the cloud background

Place the `clouds-background.svg` file in the `assets/images/` directory.

## 4. Update your layout file

Replace your existing `_layouts/default.html` file with the new one provided.

## 5. Test your changes

Commit and push these changes to your GitHub repository. GitHub Pages will automatically rebuild your site with the new sky theme.

## Optional: Customize the theme

You can customize the sky theme by editing the `sky-theme.css` file:
- Change gradient colors for a different sky appearance
- Adjust cloud animations
- Modify container transparency and shadow effects

## Troubleshooting

If the theme doesn't appear correctly:
1. Make sure all files are in the correct directories
2. Check that your `_config.yml` file is correctly configured
3. Confirm that GitHub Pages is enabled for your repository
4. Ensure the CSS paths in the layout file match your actual directory structure
