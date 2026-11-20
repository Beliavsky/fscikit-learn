! SPDX-License-Identifier: MIT
! Copyright (c) 2026 SciFort contributors
! Adapted FFT plan/radix-2 kernels from SciFort src/signal_spectral.f90.
! Changes: isolated radix-2/Bluestein paths, project kinds and require diagnostics;
! removed unrelated spectral/window/mixed-radix routines. Full license:
! LICENSES/scifort-MIT.txt. Mathematical chirp method: Bluestein (1970).
module sklearn_count_sketch_fft
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: count_sketch_fft_plan
  real(dp), parameter :: pi=acos(-1._dp)
  type :: count_sketch_fft_plan
    private
    integer :: n=0,m=0
    complex(dp), allocatable :: chirp(:),kernel(:),scratch(:)
  contains
    procedure :: initialize=>plan_initialize
    procedure :: forward=>plan_forward
    procedure :: inverse=>plan_inverse
  end type
contains
  subroutine plan_initialize(self,n)
    class(count_sketch_fft_plan), intent(out) :: self
    integer, intent(in) :: n
    integer :: j,ierr
    integer(int64) :: period,square_mod
    real(dp) :: angle
    call require(n>0,'CountSketch FFT length must be positive')
    self%n=n
    if(iand(n,n-1)==0) return
    call require(n<=(huge(n)-1)/2,'CountSketch FFT work size overflow')
    self%m=1
    do while(self%m<2*n-1)
      call require(self%m<=huge(n)/2,'CountSketch FFT work size overflow')
      self%m=2*self%m
    end do
    allocate(self%chirp(n),self%kernel(self%m),self%scratch(self%m),stat=ierr)
    call require(ierr==0,'CountSketch FFT workspace allocation failed')
    self%kernel=cmplx(0._dp,0._dp,dp); period=2_int64*int(n,int64)
    do j=0,n-1
      square_mod=modulo(int(j,int64)*int(j,int64),period)
      angle=pi*real(square_mod,dp)/real(n,dp)
      self%chirp(j+1)=cmplx(cos(angle),-sin(angle),dp)
      self%kernel(j+1)=conjg(self%chirp(j+1))
      if(j>0) self%kernel(self%m-j+1)=self%kernel(j+1)
    end do
    call radix2_fft(self%kernel)
  end subroutine
  subroutine plan_forward(self,values)
    class(count_sketch_fft_plan), intent(inout) :: self
    complex(dp), intent(inout) :: values(:)
    call require(self%n>0.and.size(values)==self%n,'CountSketch FFT plan length mismatch')
    if(self%m==0) then
      call radix2_fft(values)
    else
      self%scratch=cmplx(0._dp,0._dp,dp); self%scratch(:self%n)=values*self%chirp
      call radix2_fft(self%scratch)
      self%scratch=conjg(self%scratch*self%kernel)
      call radix2_fft(self%scratch)
      values=conjg(self%scratch(:self%n))*self%chirp/real(self%m,dp)
    end if
  end subroutine
  subroutine plan_inverse(self,values)
    class(count_sketch_fft_plan), intent(inout) :: self
    complex(dp), intent(inout) :: values(:)
    call require(self%n>0.and.size(values)==self%n,'CountSketch FFT plan length mismatch')
    values=conjg(values); call self%forward(values); values=conjg(values)/real(self%n,dp)
  end subroutine
    subroutine radix2_fft(values)
        complex(dp), intent(inout) :: values(:) !! power-of-two-length spectrum
        complex(dp) :: t, w, wm, u
        integer :: n, i, j, k, m, half, stride

        n = size(values)
        if (n <= 1) return
        j = 1
        do i = 1, n - 1
            if (i < j) then
                t = values(i)
                values(i) = values(j)
                values(j) = t
            end if
            k = n / 2
            do while (k >= 1 .and. j > k)
                j = j - k
                k = k / 2
            end do
            j = j + k
        end do
        m = 2
        do while (m <= n)
            half = m / 2
            wm = cmplx(cos(-2.0_dp*pi/real(m,dp)), &
                sin(-2.0_dp*pi/real(m,dp)),dp)
            do stride = 1, n, m
                w = (1.0_dp, 0.0_dp)
                do k = 0, half - 1
                    t = w*values(stride+k+half)
                    u = values(stride+k)
                    values(stride+k) = u + t
                    values(stride+k+half) = u - t
                    w = w*wm
                end do
            end do
            if (m == n) exit
            m = 2*m
        end do
    end subroutine radix2_fft
end module
