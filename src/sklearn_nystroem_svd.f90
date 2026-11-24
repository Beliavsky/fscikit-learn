! SPDX-License-Identifier: MIT
! Copyright (c) 2026 SciFort contributors
! Real square Jacobi SVD adapted from C:/fortran/scifort/src/linalg_svd.f90.
! Modifications: native kinds/status integration, real arithmetic, preserve small
! nonzero singular values and complete collapsed null directions. License:
! LICENSES/scifort-linalg-LICENSE.md. Used internally by sklearn Nystroem.
module sklearn_nystroem_svd
  use sklearn_common,only:dp,require
  use sklearn_ieee_math,only:ieee_product
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public::kernel_svd
contains
    subroutine kernel_svd(a, u, s, v, status, limit)
        real(dp), intent(in) :: a(:, :) !! finite input matrix with m >= n
        real(dp), intent(out) :: u(:, :) !! m-by-n orthonormal columns
        real(dp), intent(out) :: s(:) !! descending singular values
        real(dp), intent(out) :: v(:, :) !! n-by-n right singular vectors
        integer, intent(out) :: status !! linalg_status_* outcome
        integer, intent(in) :: limit !! maximum number of sweeps, >= 1
        real(dp), allocatable :: w(:, :), column(:), rcolumn(:), best_column(:)
        real(dp) :: gamma, phase
        real(dp) :: scale, na, nb, g, angle, c, sn
        real(dp) :: norm_u, candidate, best_norm, swap_sigma
        integer :: m, n, i, j, sweep, best, pivot, pass, idx
        logical :: changed, converged

        call require(size(a,1)>=size(a,2),'kernel SVD requires rows >= columns')
        m = size(a, 1)
        n = size(a, 2)
        call require(all(shape(u)==[m,n]).and.all(shape(v)==[n,n]).and.size(s)==n,'kernel SVD output shapes')
        u = 0._dp
        v = 0._dp
        s = 0.0_dp
        status = 1
        do j = 1, n
            v(j,j) = 1._dp
        end do
        scale = maxval(abs(a))
        if (scale == 0.0_dp) then
            do j = 1, n
                u(j,j) = 1._dp
            end do
            status = 0
            return
        end if
        allocate(w(m,n),column(m),rcolumn(n),best_column(m))
        w = a / scale
        converged = .false.
        do sweep = 1, limit
            changed = .false.
            do i = 1, n-1
                do j = i+1, n
                    na = sum(abs(w(:,i))**2)
                    nb = sum(abs(w(:,j))**2)
                    if (na == 0.0_dp .or. nb == 0.0_dp) cycle
                    gamma = dot_product(w(:,i),w(:,j))
                    g = abs(gamma)
                    if (g <= 32.0_dp*epsilon(1.0_dp)*sqrt(na)*sqrt(nb)) cycle
                    phase = sign(1._dp,gamma)
                    if (nb >= na) then
                        angle = 0.5_dp*atan2(2.0_dp*g,nb-na)
                    else
                        angle = -0.5_dp*atan2(2.0_dp*g,na-nb)
                    end if
                    if (abs(angle) <= epsilon(1.0_dp)) cycle
                    c = cos(angle)
                    sn = sin(angle)
                    column = c*w(:,i) - sn*phase*w(:,j)
                    w(:,j) = sn*w(:,i) + c*phase*w(:,j)
                    w(:,i) = column
                    rcolumn = c*v(:,i) - sn*phase*v(:,j)
                    v(:,j) = sn*v(:,i) + c*phase*v(:,j)
                    v(:,i) = rcolumn
                    changed = .true.
                end do
            end do
            if (.not. changed) then
                converged = .true.
                exit
            end if
        end do
        if (.not. converged) return
        do j = 1, n
            s(j) = sqrt(sum(abs(w(:,j))**2))
        end do
        do j = 1, n-1
            best = j - 1 + maxloc(s(j:n),dim=1)
            if (best == j) cycle
            swap_sigma = s(j)
            s(j) = s(best)
            s(best) = swap_sigma
            column = w(:,j)
            w(:,j) = w(:,best)
            w(:,best) = column
            rcolumn = v(:,j)
            v(:,j) = v(:,best)
            v(:,best) = rcolumn
        end do
        do j = 1, n
            column = 0._dp
            if(s(j)>0) column=w(:,j)/s(j)
            do pass = 1, 2
                do idx = 1, j-1
                    column = column-dot_product(u(:,idx),column)*u(:,idx)
                end do
            end do
            norm_u = sqrt(sum(abs(column)**2))
            if (norm_u <= 32.0_dp*epsilon(1.0_dp)) then
                s(j)=0;best_norm=-1
                do pivot=1,m
                    column=0;column(pivot)=1
                    do pass=1,2
                        do idx=1,j-1
                            column=column-dot_product(u(:,idx),column)*u(:,idx)
                        end do
                    end do
                    candidate=sqrt(sum(column**2))
                    if(candidate>best_norm) then
                        best_norm=candidate;best_column=column
                    end if
                end do
                column=best_column;norm_u=best_norm
                if(norm_u<=32*epsilon(1._dp)) return
            end if
            u(:,j) = column/norm_u
        end do
        do j=1,n
            s(j)=ieee_product(s(j),scale)
        end do
        if (.not.all(ieee_is_finite(u)).or..not.all(ieee_is_finite(v))) return
        status = 0
    end subroutine kernel_svd
end module
