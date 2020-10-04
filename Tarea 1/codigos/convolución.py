cpdef float[:, :] convolution_cython(float [:, :] input, float [:, :] mask):
  cdef int r, c, rows, cols, k_rows, k_cols, m, n, mm , nn, k_center_x, k_center_y
  cdef np.ndarray output = np.zeros((input.shape[0], input.shape[1]), dtype = np.float32)

  # tamano de la imagen, máscara y centro de la máscara
  rows = input.shape[0]
  cols = input.shape[1]
  k_rows = mask.shape[0]
  k_cols = mask.shape[1]
  k_center_x = k_cols//2
  k_center_y = k_rows//2

  # Convolucion entre "input" y "mask"
  for r in range(rows):
      for c in range(cols):
        if ((c >= k_center_x and c + k_cols - k_center_x <= cols) and 
        (r >= k_center_y  and r + k_rows - k_center_y <= rows)):
          for m in range(k_rows):
            mm = k_rows - 1 - m
            for n in range(k_cols):
              nn = k_cols - 1 - n
              ii = r + m - k_center_y
              jj = c + n - k_center_x
              output[r,c] += input[ii,jj] * mask[mm,nn]
        else:
          output[r,c] = input[r,c]
  return output