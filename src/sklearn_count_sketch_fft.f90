! SPDX-License-Identifier: MIT AND BSD-3-Clause
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
  ! Small prime butterflies below are translated from pocketfft (BSD-3-Clause),
  ! copyright (C) 2010-2022 Max-Planck-Society and (C) 2019-2020 Peter Bell.
  ! See LICENSES/pocketfft-BSD-3-Clause.txt and THIRD_PARTY_NOTICES.md.
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
    if(iand(n,n-1)==0.or.small_prime(n)) return
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
    if(small_prime(self%n)) then
      call prime_fft(values)
    else if(self%m==0) then
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
    values=conjg(values); call self%forward(values)
    ! Scale components separately: complex division can contaminate an infinite
    ! real component with a NaN imaginary component through a spurious 0*NaN.
    values=cmplx(real(values,dp)/real(self%n,dp),-aimag(values)/real(self%n,dp),dp)
  end subroutine
  logical function small_prime(n)
    integer,intent(in)::n
    integer::j
    small_prime=.false.
    if(n<3.or.n>31) return
    do j=2,int(sqrt(real(n,dp)))
      if(modulo(n,j)==0) return
    end do
    small_prime=.true.
  end function
  subroutine prime_fft(values)
    ! Single-factor versions of pocketfft cfftp pass3/5/7/11/passg. Real and
    ! imaginary arithmetic is explicit, preserving nonfinite components.
    complex(dp),intent(inout)::values(:)
    real(dp)::plus_r(size(values)/2),plus_i(size(values)/2)
    real(dp)::minus_r(size(values)/2),minus_i(size(values)/2)
    real(dp)::wr(0:size(values)-1),wi(0:size(values)-1),r0,i0,ar,ai,br,bi,dr,di
    integer::n,h,j,l,a,b
    n=size(values);h=n/2;r0=real(values(1),dp);i0=aimag(values(1))
    do j=0,n-1
      wr(j)=cos(2*pi*j/n);wi(j)=-sin(2*pi*j/n)
    end do
    if(n==3) then
      wr(1)=-.5_dp;wi(1)=-.8660254037844386467637231707529362_dp
      wr(2)=wr(1);wi(2)=-wi(1)
    else if(n==5) then
      wr(1)=.3090169943749474241022934171828191_dp;wi(1)=-.9510565162951535721164393333793821_dp
      wr(2)=-.8090169943749474241022934171828191_dp;wi(2)=-.5877852522924731291687059546390728_dp
      wr(3)=wr(2);wi(3)=-wi(2);wr(4)=wr(1);wi(4)=-wi(1)
    end if
    ar=r0;ai=i0
    do j=1,h
      plus_r(j)=real(values(j+1),dp)+real(values(n-j+1),dp)
      plus_i(j)=aimag(values(j+1))+aimag(values(n-j+1))
      minus_r(j)=real(values(j+1),dp)-real(values(n-j+1),dp)
      minus_i(j)=aimag(values(j+1))-aimag(values(n-j+1))
      ar=ar+plus_r(j);ai=ai+plus_i(j)
    end do
    values(1)=cmplx(ar,ai,dp)
    do l=1,h
      if(n<=11) then
        ar=r0;ai=i0;br=-wi(l)*minus_i(1);bi=wi(l)*minus_r(1)
        do j=1,h
          a=modulo(l*j,n);ar=ar+wr(a)*plus_r(j);ai=ai+wr(a)*plus_i(j)
          if(j==1) cycle
          br=br-wi(a)*minus_i(j);bi=bi+wi(a)*minus_r(j)
        end do
      else
        a=modulo(2*l,n)
        ar=r0+wr(l)*plus_r(1)+wr(a)*plus_r(2)
        ai=i0+wr(l)*plus_i(1)+wr(a)*plus_i(2)
        br=-wi(l)*minus_i(1)-wi(a)*minus_i(2)
        bi=wi(l)*minus_r(1)+wi(a)*minus_r(2)
        j=3
        do while(j<h)
          a=modulo(l*j,n);b=modulo(l*(j+1),n)
          ar=ar+(wr(a)*plus_r(j)+wr(b)*plus_r(j+1))
          ai=ai+(wr(a)*plus_i(j)+wr(b)*plus_i(j+1))
          br=br-(wi(a)*minus_i(j)+wi(b)*minus_i(j+1))
          bi=bi+(wi(a)*minus_r(j)+wi(b)*minus_r(j+1));j=j+2
        end do
        if(j<=h) then
          a=modulo(l*j,n);ar=ar+wr(a)*plus_r(j);ai=ai+wr(a)*plus_i(j)
          br=br-wi(a)*minus_i(j);bi=bi+wi(a)*minus_r(j)
        end if
      end if
      dr=ar+br;di=ai+bi;values(l+1)=cmplx(dr,di,dp)
      values(n-l+1)=cmplx(ar-br,ai-bi,dp)
    end do
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
                    if(k==0) then
                        t=values(stride+k+half)
                    else if(2*k==half) then
                        t=cmplx(aimag(values(stride+k+half)),-real(values(stride+k+half),dp),dp)
                    else
                        t=w*values(stride+k+half)
                    end if
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
