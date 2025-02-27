This is the source repository for my website, www.darshsrad.com.

This README describes the build process and architecture for the website.
If you're familiar with GitHub actions and the general concept
of a static site generator, you may want to skip straight to the
[workflow](./.github/worfklows/pages.yml)
that actually builds and deploys the site.

The discussion below assumes Linux as the host system. Ubuntu and
Arch have been tested on pretty extensively, but you should be able
to adapt everything to your distro of choice.

## Prerequisites

From the system package manager, you'll need Python, Ruby, and curl.
Nothing else here should require `sudo` or any system-level changes.

### Python Setup

Follow the initial setup instructions in [\_nb](./nb/README.md).

### Ruby Setup

Install the `bundler` Ruby gem, and run `bundle install` in the repository
root.

## Building the Site

There are two preliminary steps to prepare Markdown sources and other assets,
followed by the main Jekyll build.

### Generating Notebooks

Run `_scripts/generate-nb.sh`. This needs to be redone whenever the notebook
content in `_nb` changes.

### Fetching Third-Party Assets

Run `_scripts/fetch-assets.sh`. This needs to be redone only when the third-party
assets themselves change, which should be seldom.

### Running the Jekyll Build

Run `bundle exec jekyll build`.

## Testing

As an alternative to `jekyll build`, `jekyll serve` can be used to run a local
web server, allowing the site to be checked in a browser.
This is the suggested invocation:

    bundle exec jekyll serve --host 0.0.0.0

which allows the site to be viewed at `http://localhost:4000` on any computer
on the local network.
This will automatically rebuild the site whenever any Jekyll source file is modified,
but it will not catch changes in `_nb` because Jekyll doesn't process that
directly.
`_scripts/generate-nb.sh` will need to be rerun to propagate changes from `_nb`.

## Architecture and Organization

This is a pretty vanilla [Jekyll](https://jekyllrb.com/) site,
using the [Just the Docs](https://just-the-docs.com/) theme.
The site is deployed to GitHub Pages via a GitHub action,
so any push to the master branch will automatically update the main website
within a few minutes, if the action completes successfully.

### Same-Source Policy

All web assets that are necessary for correct rendering of the site,
including stylesheets, scripts, fonts, and embedded images,
must be served from the site itself.
Thus, all such assets must be either tracked in version control,
generated during the build, or fetched from third-party sources
during the build.
This rule makes the site self-contained and allows many site-breaking
errors to be caught during the build before deployment.

As a consequence, no third-party asset whose copyright license doesn't allow such
redistribution can be used on this site.

### Third-Party Assets

The policy in this repository is that third-party assets should not be tracked
directly in Git, but should be fetched during the build.
While this results in slightly more web requests during the deployment process,
in practice it's not an issue because the website isn't redeployed very frequently,
and assets rarely need to be refetched on a single repository clone for testing
locally.
This rule helps keep the overall architecture and build process of the website
clean and easy to follow.

Exceptions can be made to this policy on a case-by-case basis.

### Generated Assets

For derived assets, i.e., those which can be generated programmatically,
we prefer to track the sources
rather than the assets themselves,
with an exception for "major" works that are central to the site,
like mathematical art pieces.
The major works should be tracked in version control and not regenerated.

Derived assets that require very resource-intensive computations would also
generally be excepted from this rule.

### Jupyter Notebooks

Jupyter notebooks are be published from the `_nb` directory via a process
documented in the [`README.md`](./_nb/README.md) file in that directory.
