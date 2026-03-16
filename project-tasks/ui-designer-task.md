## UI Designer Task Context

You are the **UI Designer** for DocBot enhancement. Your mission is to make the UI production-ready, top-of-the-class, and easily usable.

### Current State Analysis

**Frontend** (`/src/app/page.tsx`):
- Dark theme with purple/blue gradients
- Sidebar for options (document upload, persona selection, toggles)
- Main chat area with message bubbles
- Basic Lucide icons
- Functional but basic aesthetics

**Issues Identified**:
1. Generic gradient backgrounds
2. Basic toggle switches
3. Limited visual hierarchy
4. No smooth animations/transitions
5. Text could be more readable
6. No empty states or helpful prompts
7. Upload area is plain

### Your Task

**Primary Objectives**:

1. **Modernize Visual Design**:
   - Refine the color scheme (professional, not generic purple)
   - Improve typography and spacing
   - Add depth through shadows and layers
   - Create visual hierarchy

2. **Enhance Usability**:
   - Better file upload area with drag-and-drop
   - Clearer persona selection (cards vs dropdown)
   - Improved toggle switches
   - Better loading states
   - Error states that are helpful

3. **Polish Interactions**:
   - Smooth transitions on hover/focus
   - Micro-animations for feedback
   - Scroll behaviors
   - Input field improvements

4. **Accessibility**:
   - Proper contrast ratios
   - Focus states
   - Keyboard navigation hints

### Deliverable

Provide complete updated:
1. `/src/app/page.tsx` - Main UI component with enhancements
2. `/src/app/globals.css` - Any additional styles needed
3. Maintain all current functionality (file upload, chat, personas)

### Notes
- Keep the dark theme (it works well for document apps)
- Maintain the current tech stack (React, Tailwind, Lucide)
- Ensure responsive design works on tablet+
- Keep message bubbles easily distinguishable
