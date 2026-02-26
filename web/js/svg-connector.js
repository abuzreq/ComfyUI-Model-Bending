// SVG Connector Module
class SVGConnector {
    constructor() {
        this.overlay = null;
        this.poly = null;
        this.observers = [];
    }

    /**
     * Link two SVG elements with a visual connector
     * @param {Element} svgX - Source SVG element
     * @param {Element} svgY - Target SVG element
     * @param {Object} options - Configuration options
     * @returns {Object} Connector object with update and remove methods
     */
    linkSVGs(svgX, svgY, options = {}) {
        const {
            insetTop = 24,
            insetBottom = 48,
            color = "rgb(232, 245, 233)",
            alpha = 0.3,
            strokeAlpha = 0.55,
            strokeWidth = 2
        } = options;

        console.log(`Linking SVGs: ${svgX} → ${svgY} with color ${color}`);
        
        // Remove existing overlay
        this.removeOverlay();
        
        // Create new overlay
        this.createOverlay();
        
        // Create polygon
        this.poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        this.poly.setAttribute("fill", Utils.rgba(color, alpha));
        this.poly.setAttribute("stroke", Utils.rgba(color, strokeAlpha));
        this.poly.setAttribute("stroke-width", strokeWidth);
        this.poly.setAttribute("stroke-dasharray", 10);
        this.overlay.appendChild(this.poly);

        const computeAndDraw = () => {
            if (!svgX || !svgY) return;
            
            const xr = svgX.getBoundingClientRect();
            const yr = svgY.getBoundingClientRect();
            
            // If Y is hidden, skip drawing
            const hidden = yr.width === 0 && yr.height === 0;
            if (hidden) {
                this.poly.setAttribute("points", "");
                return;
            }

            // Calculate connector points
            const topLeft = [xr.left + insetTop, xr.bottom];
            const topRight = [xr.right - insetTop, xr.bottom];
            const bottomRight = [yr.right - insetBottom, yr.top];
            const bottomLeft = [yr.left + insetBottom, yr.top];

            const pts = [topLeft, topRight, bottomRight, bottomLeft]
                .map(([x, y]) => `${x},${y}`)
                .join(" ");
            
            this.poly.setAttribute("points", pts);
        };

        // Set up event listeners
        const onScroll = () => computeAndDraw();
        const onResize = () => computeAndDraw();
        
        window.addEventListener("scroll", onScroll, { passive: true });
        window.addEventListener("resize", onResize);

        // Set up ResizeObserver if available
        let resizeObserver = null;
        if ("ResizeObserver" in window) {
            resizeObserver = new ResizeObserver(computeAndDraw);
            resizeObserver.observe(svgX);
            resizeObserver.observe(svgY);
        }

        // Initial draw
        requestAnimationFrame(computeAndDraw);

        // Store cleanup functions
        this.observers.push({
            onScroll,
            onResize,
            resizeObserver
        });

        return {
            update: computeAndDraw,
            remove: () => this.remove()
        };
    }

    /**
     * Create overlay SVG element
     */
    createOverlay() {
        this.overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        this.overlay.id = "svg-connector-overlay";
        
        Object.assign(this.overlay.style, {
            position: "fixed",
            inset: "0",
            width: "100vw",
            height: "100vh",
            pointerEvents: "none",
            zIndex: -1,
        });
        
        document.body.appendChild(this.overlay);
    }

    /**
     * Remove overlay and clean up
     */
    removeOverlay() {
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
        }
    }

    /**
     * Remove connector and clean up all resources
     */
    remove() {
        // Remove event listeners
        this.observers.forEach(({ onScroll, onResize, resizeObserver }) => {
            window.removeEventListener("scroll", onScroll);
            window.removeEventListener("resize", onResize);
            if (resizeObserver) resizeObserver.disconnect();
        });
        
        this.observers = [];
        
        // Remove polygon
        if (this.poly) {
            this.poly.remove();
            this.poly = null;
        }
        
        // Remove overlay if empty
        if (this.overlay && !this.overlay.querySelector("polygon")) {
            this.removeOverlay();
        }
    }
}

// Global instance
const svgConnector = new SVGConnector(); 