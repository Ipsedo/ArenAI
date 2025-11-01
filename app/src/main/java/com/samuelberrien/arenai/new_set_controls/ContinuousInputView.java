package com.samuelberrien.arenai.new_set_controls;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.IntDef;
import androidx.annotation.Nullable;

import com.samuelberrien.arenai.R;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

public class ContinuousInputView extends View {
    public static final int FROM_LEFT = 0;
    public static final int FROM_RIGHT = 1;
    public static final int FROM_TOP = 2;
    public static final int FROM_BOTTOM = 3;

    @IntDef({FROM_LEFT, FROM_RIGHT, FROM_TOP, FROM_BOTTOM})
    @Retention(RetentionPolicy.SOURCE)
    public @interface GrowFrom {}

    private final Paint backgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint strokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final RectF contentRect = new RectF();
    private final RectF fillRect = new RectF();

    private int fixedRectColor = 0xFFEFEFEF;
    private int fixedRectStrokeColor = 0xFF222222;
    private float fixedRectStrokeWidthPx;

    private int fillRectColor = 0xFF3F51B5;

    @GrowFrom
    private int growFrom = FROM_LEFT;

    private float value = 0f;   // valeur courante
    private float max = 1f;     // maximum (si =1, value est déjà normalisée)

    private int inputKindRefId;

    private ContinuousInputListener continuousInputListener;

    public ContinuousInputView(Context context) {
        super(context);
        init(context, null);
    }

    public ContinuousInputView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init(context, attrs);
    }

    public ContinuousInputView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init(context, attrs);
    }

    private void init(Context context, @Nullable AttributeSet attrs) {
        fixedRectStrokeWidthPx = dp(2);

        if (attrs != null) {
            try (TypedArray a = getContext().obtainStyledAttributes(attrs, R.styleable.ContinuousInputView)) {
                try {
                    fixedRectColor = a.getColor(R.styleable.ContinuousInputView_fixedRectColor, fixedRectColor);
                    fixedRectStrokeColor = a.getColor(R.styleable.ContinuousInputView_fixedRectStrokeColor, fixedRectStrokeColor);
                    fixedRectStrokeWidthPx = a.getDimension(R.styleable.ContinuousInputView_fixedRectStrokeWidth, fixedRectStrokeWidthPx);

                    fillRectColor = a.getColor(R.styleable.ContinuousInputView_fillRectColor, fillRectColor);

                    growFrom = a.getInt(R.styleable.ContinuousInputView_growFrom, FROM_LEFT);

                    value = a.getFloat(R.styleable.ContinuousInputView_value, 0f);
                    max = a.getFloat(R.styleable.ContinuousInputView_max, 1f);

                    inputKindRefId = a.getInt(R.styleable.ContinuousInputView_input, 0);
                } finally {
                    a.recycle();
                }
            }
        }

        backgroundPaint.setStyle(Paint.Style.FILL);
        backgroundPaint.setColor(fixedRectColor);

        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setColor(fillRectColor);

        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeWidth(fixedRectStrokeWidthPx);
        strokePaint.setColor(fixedRectStrokeColor);

        setWillNotDraw(false);

        continuousInputListener = new ContinuousInputListener(context, Integer.toString(inputKindRefId));

        setOnClickListener(v -> {
            // TODO listen input and update shared pref
        });
    }

    private float dp(float v) {
        return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        // Taille par défaut si wrap_content
        int desiredW = (int) dp(160);
        int desiredH = (int) dp(24);

        int w = resolveSize(desiredW, widthMeasureSpec);
        int h = resolveSize(desiredH, heightMeasureSpec);

        setMeasuredDimension(w, h);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // Aire intérieure tenant compte des padding
        contentRect.set(
                getPaddingLeft(),
                getPaddingTop(),
                getWidth() - getPaddingRight(),
                getHeight() - getPaddingBottom()
        );

        // Fond du rectangle fixe
        backgroundPaint.setColor(fixedRectColor);
        canvas.drawRect(contentRect, backgroundPaint);

        // Zone dans laquelle le remplissage doit rester (à l'intérieur du trait)
        float inset = fixedRectStrokeWidthPx / 2f;
        @SuppressLint("DrawAllocation") RectF inner = new RectF(
                contentRect.left + inset,
                contentRect.top + inset,
                contentRect.right - inset,
                contentRect.bottom - inset
        );

        // Calcul du pourcentage (0..1)
        float pct = value / max;
        if (pct < 0f) pct = 0f;
        if (pct > 1f) pct = 1f;

        // Calcul du rectangle de remplissage selon le côté choisi
        fillPaint.setColor(fillRectColor);

        switch (growFrom) {
            case FROM_LEFT: {
                float w = inner.width() * pct;
                fillRect.set(inner.left, inner.top, inner.left + w, inner.bottom);
                break;
            }
            case FROM_RIGHT: {
                float w = inner.width() * pct;
                fillRect.set(inner.right - w, inner.top, inner.right, inner.bottom);
                break;
            }
            case FROM_TOP: {
                float h = inner.height() * pct;
                fillRect.set(inner.left, inner.top, inner.right, inner.top + h);
                break;
            }
            case FROM_BOTTOM:
            default: {
                float h = inner.height() * pct;
                fillRect.set(inner.left, inner.bottom - h, inner.right, inner.bottom);
                break;
            }
        }

        // Dessine le remplissage (à l'intérieur du contour)
        if (pct > 0f) {
            canvas.drawRect(fillRect, fillPaint);
        }

        // Dessine le contour par-dessus pour que le remplissage reste "sous" le contour visuellement
        strokePaint.setStrokeWidth(fixedRectStrokeWidthPx);
        strokePaint.setColor(fixedRectStrokeColor);
        canvas.drawRect(contentRect, strokePaint);
    }

    @Override
    public boolean onGenericMotionEvent(MotionEvent event) {
        if (continuousInputListener.onGenericMotion(this, event)) {
            value = continuousInputListener.getValue();
            requestLayout();
            return true;
        }

        return super.onGenericMotionEvent(event);
    }
}
