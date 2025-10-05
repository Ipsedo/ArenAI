//
// Created by samuel on 28/03/2023.
//

#ifndef PHYVR_VIEW_H
#define PHYVR_VIEW_H

#include <map>
#include <string>
#include <vector>

#include <android/configuration.h>

class View {
public:
    View(int screen_width, int screen_height);
    void add_dimen(const std::string &dimen_name, int dimen_dp);
    float get_pixel(AConfiguration *config, const std::string &dimen_name);
    virtual float get_width() = 0;
    virtual float get_height() = 0;
    virtual float get_margin() = 0;

private:
    std::map<std::string, int> dimens_dps;

protected:
    const int screen_width, screen_height;

    virtual void set_width(float width) = 0;
    virtual void set_height(float height) = 0;
};

class LinearLayout : public View {
public:
    enum ORIENTATION { HORIZONTAL = 0, VERTICAL = 1 };

    LinearLayout(int screen_width, int screen_height, LinearLayout::ORIENTATION orientation);
    void add_view(const std::shared_ptr<View> &view);
    void build();

    float get_width() override;

    float get_height() override;

    float get_margin() override;

protected:
    void set_width(float width) override;

    void set_height(float height) override;

private:
    std::vector<std::shared_ptr<View>> views;
    LinearLayout::ORIENTATION orientation;
};

class CornerLayout : public View {
public:
    enum CORNER { LEFT_TOP = 0, RIGHT_TOP = 1, RIGHT_BOTTOM = 2, LEFT_BOTTOM = 3 };
    CornerLayout(int screen_width, int screen_height);
    void add_view(const std::shared_ptr<View> &view, CornerLayout::CORNER corner);
    void build();

    float get_width() override;

    float get_height() override;

    float get_margin() override;

protected:
    void set_width(float width) override;

    void set_height(float height) override;

private:
    std::map<CornerLayout::CORNER, std::shared_ptr<View>> views;
};

#endif// PHYVR_VIEW_H
