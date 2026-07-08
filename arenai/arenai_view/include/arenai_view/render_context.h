//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_RENDER_CONTEXT_H
#define ARENAI_RENDER_CONTEXT_H

namespace arenai::view {

    class AbstractRenderContext {
    public:
        virtual ~AbstractRenderContext() = default;

        virtual void make_current() = 0;
        virtual void release_current() = 0;
    };

}// namespace arenai::view

#endif// ARENAI_RENDER_CONTEXT_H
