! Imputation contracts: sklearn 1.6.1 impute/_knn.py and _base.py MissingIndicator.
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
module sklearn_knn_impute
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: knn_imputer,missing_indicator
  type :: missing_indicator
    character(16) :: features='missing-only'
    logical :: error_on_new=.true.
    integer :: n_features_in_=0
    integer, allocatable :: features_(:)
  contains
    procedure :: fit=>indicator_fit
    procedure :: transform=>indicator_transform
    procedure :: fit_transform=>indicator_fit_transform
  end type
  type :: knn_imputer
    integer :: n_neighbors=5,n_features_in_=0
    character(16) :: weights='uniform'
    logical :: keep_empty_features=.false.,add_indicator=.false.
    logical, allocatable :: valid_mask_(:)
    type(missing_indicator) :: indicator_
    real(dp), allocatable, private :: training_(:,:),means_(:)
  contains
    procedure :: fit=>knn_imputer_fit
    procedure :: transform=>knn_imputer_transform
    procedure :: fit_transform=>knn_imputer_fit_transform
  end type
contains
  subroutine impute_check(x)
    real(dp), intent(in) :: x(:,:)
    call require(size(x,1)>0.and.size(x,2)>0,'imputation X must be nonempty')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'imputation X contains infinity')
  end subroutine

  subroutine indicator_fit(self,x)
    class(missing_indicator), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    logical, allocatable :: missing(:)
    integer :: j
    call impute_check(x)
    call require(self%features=='missing-only'.or.self%features=='all','unknown MissingIndicator features mode')
    self%n_features_in_=size(x,2); missing=any(ieee_is_nan(x),dim=1)
    if(self%features=='all') missing=.true.
    self%features_=pack([(j,j=1,size(x,2))],missing)
  end subroutine

  function indicator_transform(self,x) result(mask)
    class(missing_indicator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    logical, allocatable :: mask(:,:),selected(:)
    integer :: j
    call require(allocated(self%features_),'MissingIndicator is not fitted')
    call impute_check(x)
    call require(size(x,2)==self%n_features_in_,'MissingIndicator feature count mismatch')
    if(self%features=='missing-only'.and.self%error_on_new) then
      allocate(selected(size(x,2))); selected=.false.; selected(self%features_)=.true.
      do j=1,size(x,2)
        if(.not.selected(j)) call require(.not.any(ieee_is_nan(x(:,j))),'MissingIndicator new missing feature')
      end do
    end if
    mask=ieee_is_nan(x(:,self%features_))
  end function

  function indicator_fit_transform(self,x) result(mask)
    class(missing_indicator), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    logical, allocatable :: mask(:,:)
    call self%fit(x); mask=self%transform(x)
  end function

  subroutine knn_imputer_fit(self,x)
    class(knn_imputer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: j,i,n
    call impute_check(x)
    call require(self%n_neighbors>0,'KNNImputer n_neighbors must be positive')
    call require(self%weights=='uniform'.or.self%weights=='distance','unknown KNNImputer weights')
    self%training_=x; self%n_features_in_=size(x,2)
    self%valid_mask_=any(.not.ieee_is_nan(x),dim=1)
    self%means_=spread(0._dp,1,size(x,2))
    do j=1,size(x,2)
      n=0
      do i=1,size(x,1)
        if(ieee_is_nan(x(i,j))) cycle
        n=n+1; self%means_(j)=self%means_(j)+x(i,j)
      end do
      if(n>0) self%means_(j)=self%means_(j)/n
    end do
    self%indicator_%features='missing-only'; self%indicator_%error_on_new=.false.
    call self%indicator_%fit(x)
  end subroutine

  function knn_imputer_transform(self,x) result(z)
    class(knn_imputer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),values(:,:),distances(:),donor_values(:),donor_distances(:),w(:)
    logical, allocatable :: defined(:),indicators(:,:)
    integer, allocatable :: columns(:)
    real(dp) :: squared,temp
    integer :: i,j,k,r,n,p,overlap,donors,selected,width,position
    call require(allocated(self%training_),'KNNImputer is not fitted')
    call impute_check(x)
    call require(size(x,2)==self%n_features_in_,'KNNImputer feature count mismatch')
    call require(self%n_neighbors>0,'KNNImputer n_neighbors must be positive')
    call require(self%weights=='uniform'.or.self%weights=='distance','unknown KNNImputer weights')
    n=size(self%training_,1); p=size(x,2); values=x
    allocate(distances(n),defined(n),donor_values(n),donor_distances(n),w(n))
    do i=1,size(x,1)
      if(.not.any(ieee_is_nan(x(i,:)).and.self%valid_mask_)) cycle
      ! Always measure from the original receiver, before any cells are imputed.
      do r=1,n
        squared=0; overlap=0
        do k=1,p
          if(ieee_is_nan(x(i,k)).or.ieee_is_nan(self%training_(r,k))) cycle
          squared=squared+(x(i,k)-self%training_(r,k))**2; overlap=overlap+1
        end do
        defined(r)=overlap>0; distances(r)=0
        if(overlap>0) distances(r)=sqrt(squared*real(p,dp)/overlap)
      end do
      do j=1,p
        if(.not.self%valid_mask_(j)) cycle
        if(.not.ieee_is_nan(x(i,j))) cycle
        donors=0
        do r=1,n
          if(.not.defined(r).or.ieee_is_nan(self%training_(r,j))) cycle
          donors=donors+1; donor_values(donors)=self%training_(r,j); donor_distances(donors)=distances(r)
        end do
        if(donors==0) then
          values(i,j)=self%means_(j); cycle
        end if
        ! Stable distance ordering: tied donors retain their training row order.
        do k=2,donors
          temp=donor_values(k); squared=donor_distances(k); position=k-1
          do while(position>=1)
            if(donor_distances(position)<=squared) exit
            donor_distances(position+1)=donor_distances(position)
            donor_values(position+1)=donor_values(position); position=position-1
          end do
          donor_distances(position+1)=squared; donor_values(position+1)=temp
        end do
        selected=min(self%n_neighbors,donors); w(:selected)=1
        if(self%weights=='distance') then
          if(any(donor_distances(:selected)==0)) then
            w(:selected)=merge(1._dp,0._dp,donor_distances(:selected)==0)
          else
            ! Scaling by the nearest distance avoids reciprocal overflow.
            w(:selected)=donor_distances(1)/donor_distances(:selected)
          end if
        end if
        values(i,j)=sum(w(:selected)*donor_values(:selected))/sum(w(:selected))
      end do
    end do
    if(self%keep_empty_features) then
      columns=[(j,j=1,p)]
      do j=1,p
        if(.not.self%valid_mask_(j)) values(:,j)=0
      end do
    else
      columns=pack([(j,j=1,p)],self%valid_mask_)
    end if
    width=size(columns)
    if(self%add_indicator) then
      indicators=self%indicator_%transform(x)
      allocate(z(size(x,1),width+size(indicators,2)))
      z(:,:width)=values(:,columns); z(:,width+1:)=merge(1._dp,0._dp,indicators)
    else
      z=values(:,columns)
    end if
  end function

  function knn_imputer_fit_transform(self,x) result(z)
    class(knn_imputer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
