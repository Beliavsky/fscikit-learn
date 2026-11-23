! sklearn 1.6.1 _polynomial.py and _csr_polynomial_expansion.pyx (BSD).
! See THIRD_PARTY_NOTICES.md. Original Fortran adapters/counting are MIT.
module sklearn_polynomial
  use sklearn_common, only: dp,require,check_x
  use sklearn_sparse, only: csr_matrix
  use sklearn_feature_extraction, only: feature_string
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_stateless_preprocessing, only: sorted_sparse
  use sklearn_ieee_math, only: ieee_product
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: polynomial_features
  type, extends(dense_transformer) :: polynomial_features
    integer(int64) :: degree=2,min_degree=0,n_output_features_=0,min_degree_=0,max_degree_=0
    integer :: n_features_in_=0
    logical :: include_bias=.true.,interaction_only=.false.,fitted_bias_=.true.,fitted_interaction_=.false.
    type(feature_string), allocatable :: feature_names_in_(:)
  contains
    procedure :: is_fitted=>bridge_fitted
    procedure :: bridge_features=>bridge_features
    procedure :: fit_data=>poly_fit
    procedure, private :: fit_sparse=>poly_fit_sparse
    generic :: fit=>fit_data,fit_sparse
    procedure :: map_data=>poly_dense
    procedure, private :: transform_sparse=>poly_sparse
    generic :: transform=>map_data,transform_sparse
    procedure, private :: fit_transform_dense=>poly_fit_transform,fit_transform_sparse=>poly_fit_transform_sparse
    generic :: fit_transform=>fit_transform_dense,fit_transform_sparse
    procedure :: feature_names=>poly_names
    procedure :: get_feature_names_out=>poly_names
    procedure :: get_powers=>poly_powers
    procedure :: powers_=>poly_powers
  end type
  interface polynomial_features
    module procedure polynomial_array
  end interface
contains
  integer(int64) function gcd(a,b) result(value)
    integer(int64), intent(in) :: a,b
    integer(int64) :: x,y,t
    x=a;y=b
    do while(y/=0)
      t=modulo(x,y);x=y;y=t
    end do
    value=x
  end function
  integer(int64) function multiply_count(a,b) result(value)
    integer(int64), intent(in) :: a,b
    value=0
    if(a==0.or.b==0) return
    call require(a<=huge(a)/b,'polynomial feature count exceeds int64 range');value=a*b
  end function
  integer(int64) function add_count(a,b) result(value)
    integer(int64), intent(in) :: a,b
    call require(a<=huge(a)-b,'polynomial feature count exceeds int64 range');value=a+b
  end function
  integer(int64) function choose(n,k) result(value)
    integer(int64), intent(in) :: n,k
    integer(int64) :: i,steps,a,b,g
    value=0
    if(k<0.or.k>n) return
    steps=min(k,n-k);value=1;i=0
    do while(i<steps)
      i=i+1;a=n-steps+i;b=i;g=gcd(a,b);a=a/g;b=b/g
      g=gcd(value,b);value=value/g;b=b/g
      call require(b==1,'polynomial internal binomial division failure');value=multiply_count(value,a)
    end do
  end function
  integer(int64) function term_count(p,d,interaction) result(value)
    integer(int64), intent(in) :: p,d
    logical, intent(in) :: interaction
    if(interaction) then
      value=choose(p,d)
    else
      value=choose(add_count(p,d-1),min(d,p-1))
    end if
  end function
  integer(int64) function output_count(p,lo,hi,interaction,bias) result(value)
    integer(int64), intent(in) :: p,lo,hi
    logical, intent(in) :: interaction,bias
    integer(int64) :: first,last,d,width,j,base,part
    value=merge(1_int64,0_int64,bias);first=max(1_int64,lo);last=hi
    if(interaction) last=min(last,p)
    if(last<first) return
    if(interaction) then
      d=first
      do
        value=add_count(value,choose(p,d));if(d==last) exit
        d=d+1
      end do
    else
      ! Vandermonde difference avoids large intermediate binomials and an
      ! O(degree) loop when fitting a narrow range of very high degrees.
      width=last-first+1;base=add_count(p,first-1);j=1
      do
        part=multiply_count(choose(width,j),choose(base,p-j));value=add_count(value,part)
        if(j==min(p,width)) exit
        j=j+1
      end do
    end if
  end function
  subroutine controls(self)
    class(polynomial_features), intent(in) :: self
    call require(self%min_degree>=0.and.self%degree>=self%min_degree,'invalid polynomial degree range')
    call require(self%degree/=0.or.self%include_bias,'zero polynomial degree without bias has no output features')
  end subroutine
  subroutine fitted(self,width)
    class(polynomial_features), intent(in) :: self
    integer, optional, intent(in) :: width
    call require(self%n_features_in_>0,'PolynomialFeatures is not fitted')
    call require(self%degree==self%max_degree_.and.self%min_degree==self%min_degree_.and. &
      (self%include_bias.eqv.self%fitted_bias_).and.(self%interaction_only.eqv.self%fitted_interaction_), &
      'PolynomialFeatures parameters changed; refit is required')
    if(present(width)) call require(width==self%n_features_in_,'PolynomialFeatures feature count mismatch')
  end subroutine
  subroutine record_fit(self,p,names)
    class(polynomial_features), intent(inout) :: self
    integer, intent(in) :: p
    type(feature_string), optional, intent(in) :: names(:)
    integer :: i
    call controls(self)
    self%n_output_features_=output_count(int(p,int64),self%min_degree,self%degree,self%interaction_only,self%include_bias)
    self%n_features_in_=p;self%min_degree_=self%min_degree;self%max_degree_=self%degree
    self%fitted_bias_=self%include_bias;self%fitted_interaction_=self%interaction_only
    if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
    if(present(names)) then
      call require(size(names)==p,'PolynomialFeatures fit name count mismatch')
      do i=1,p
        call require(allocated(names(i)%text),'PolynomialFeatures feature name is unallocated')
      end do
      self%feature_names_in_=names
    end if
  end subroutine
  subroutine poly_fit(self,x,feature_names)
    class(polynomial_features), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    call check_x(x);call record_fit(self,size(x,2),feature_names)
  end subroutine
  subroutine sparse_data(x)
    type(csr_matrix), intent(in) :: x
    call x%validate();call require(x%n_rows>0.and.x%n_cols>0,'polynomial input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'polynomial input contains NaN or infinity')
  end subroutine
  subroutine poly_fit_sparse(self,x,feature_names)
    class(polynomial_features), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(feature_string), optional, intent(in) :: feature_names(:)
    call sparse_data(x);call record_fit(self,x%n_cols,feature_names)
  end subroutine
  subroutine allocate_matrix(a,n,p)
    real(dp), allocatable, intent(out) :: a(:,:)
    integer, intent(in) :: n
    integer(int64), intent(in) :: p
    integer :: status
    call require(p<=huge(p)/8/max(1,n),'polynomial dense allocation exceeds addressable range')
    allocate(a(n,p),stat=status);call require(status==0,'polynomial dense allocation failed')
  end subroutine
  function poly_dense(self,x) result(z)
    class(polynomial_features), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),previous(:,:),current(:,:)
    integer(int64), allocatable :: starts(:),next(:)
    integer(int64) :: degree,last,p,offset,at,first,width,number,c,i,j
    call fitted(self,size(x,2));call check_x(x);p=size(x,2,kind=int64)
    call allocate_matrix(z,size(x,1),self%n_output_features_);offset=1
    if(self%include_bias) then
      z(:,1)=1;offset=2
    end if
    last=self%max_degree_;if(self%interaction_only) last=min(last,p)
    if(last<max(1_int64,self%min_degree_)) return
    previous=x
    if(self%min_degree_<=1) then
      z(:,offset:offset+p-1)=x;offset=offset+p
    end if
    if(last==1) return
    allocate(starts(p+1),next(p+1));starts=[(i,i=1,p+1)];degree=1
    do while(degree<last)
      degree=degree+1;number=term_count(p,degree,self%interaction_only)
      call allocate_matrix(current,size(x,1),number);at=1;next=1
      do j=1,p
        next(j)=at;first=starts(j);if(self%interaction_only) first=starts(j+1)
        width=size(previous,2,kind=int64)-first+1
        if(width<=0) then
          next(j:)=at;exit
        end if
        do c=0,width-1
          do i=1,size(x,1,kind=int64)
            ! sklearn's dense suffix reuse preserves multiplication association.
            current(i,at+c)=ieee_product(previous(i,first+c),x(i,j))
          end do
        end do
        at=at+width
      end do
      next(p+1)=at;call require(at==number+1,'polynomial internal dense ordering failure')
      if(degree>=self%min_degree_) then
        z(:,offset:offset+number-1)=current;offset=offset+number
      end if
      call move_alloc(current,previous);starts=next
    end do
    call require(offset==self%n_output_features_+1,'polynomial internal output count failure')
  end function
  subroutine initial_power(power,degree,interaction)
    integer(int64), intent(out) :: power(:)
    integer(int64), intent(in) :: degree
    logical, intent(in) :: interaction
    power=0
    if(interaction) then
      power(:degree)=1
    else
      power(1)=degree
    end if
  end subroutine
  subroutine next_power(power,interaction,done)
    integer(int64), intent(inout) :: power(:)
    logical, intent(in) :: interaction
    logical, intent(out) :: done
    integer :: i
    integer(int64) :: tail
    done=.false.
    do i=size(power)-1,1,-1
      if(power(i)==0) cycle
      if(interaction) then
        if(power(i+1)/=0) cycle
        tail=count(power(i+1:)==1,kind=int64);power(i:)=0;power(i+1:i+1+tail)=1
      else
        tail=sum(power(i+1:))+1;power(i)=power(i)-1;power(i+1:)=0;power(i+1)=tail
      end if
      return
    end do
    done=.true.
  end subroutine
  integer(int64) function column_rank(columns,power,p,degree,interaction) result(rank)
    integer, intent(in) :: columns(:)
    integer(int64), intent(in) :: power(:),p,degree
    logical, intent(in) :: interaction
    integer(int64) :: total,previous,position,chosen,left,right,repetition
    integer :: j
    rank=0
    if(degree==1) then
      do j=1,size(power)
        if(power(j)==1) then
          rank=int(columns(j),int64)-1;return
        end if
      end do
    end if
    if(p==1) return
    total=p;if(.not.interaction) total=add_count(p,degree-1)
    previous=-1;position=0
    do j=1,size(power)
      repetition=0
      do while(repetition<power(j))
        repetition=repetition+1;position=position+1;chosen=int(columns(j),int64)-1
        if(.not.interaction) chosen=chosen+position-1
        if(chosen>previous+1) then
          left=choose(total-previous-1,degree-position+1);right=choose(total-chosen,degree-position+1)
          rank=add_count(rank,left-right)
        end if
        previous=chosen
      end do
    end do
  end function
  subroutine push(indices,data,used,column,value)
    integer, allocatable, intent(inout) :: indices(:)
    real(dp), allocatable, intent(inout) :: data(:)
    integer, intent(inout) :: used
    integer(int64), intent(in) :: column
    real(dp), intent(in) :: value
    integer, allocatable :: grown_indices(:)
    real(dp), allocatable :: grown_data(:)
    integer :: capacity
    call require(used<huge(1)-1,'polynomial CSR stored count exceeds native index range')
    if(used==size(data)) then
      capacity=int(min(2*int(size(data),int64),int(huge(1)-1,int64)))
      allocate(grown_indices(capacity),grown_data(capacity));grown_indices(:used)=indices(:used);grown_data(:used)=data(:used)
      call move_alloc(grown_indices,indices);call move_alloc(grown_data,data)
    end if
    used=used+1;indices(used)=int(column);data(used)=value
  end subroutine
  function poly_sparse(self,x) result(z)
    class(polynomial_features), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: z,canonical
    integer, allocatable :: indices(:),columns(:)
    real(dp), allocatable :: data(:),values(:)
    logical, allocatable :: stored(:)
    integer(int64), allocatable :: power(:)
    integer :: row,left,right,m,j,used
    integer(int64) :: degree,last,offset,column,repetition
    real(dp) :: value,largest
    logical :: done,keep,overflow_possible
    call fitted(self,x%n_cols);call sparse_data(x)
    call require(self%n_output_features_<=huge(1),'polynomial CSR feature count exceeds native index range')
    if(self%n_output_features_==merge(1_int64,0_int64,self%include_bias)) then
      z%n_rows=x%n_rows;z%n_cols=int(self%n_output_features_)
      allocate(z%indptr(x%n_rows+1),z%indices(z%n_rows*z%n_cols),z%data(z%n_rows*z%n_cols))
      z%indices=1;z%data=1
      do row=1,x%n_rows
        z%indptr(row)=1+(row-1)*z%n_cols
      end do
      z%indptr(x%n_rows+1)=1+x%n_rows*z%n_cols
      return
    end if
    canonical=sorted_sparse(x,.true.)
    call require(all(ieee_is_finite(canonical%data)),'polynomial canonicalized input exceeds finite working range')
    z%n_rows=x%n_rows;z%n_cols=int(self%n_output_features_)
    allocate(z%indptr(x%n_rows+1),indices(16),data(16));used=0;z%indptr(1)=1
    do row=1,x%n_rows
      if(self%include_bias) call push(indices,data,used,1_int64,1._dp)
      left=canonical%indptr(row);right=canonical%indptr(row+1)-1;m=right-left+1
      if(m>0) then
        ! scipy's CSC sparse multiply visits the union of row indices. After
        ! overflow, an IMPLICIT zero can therefore yield a stored NaN. Only
        ! that extreme path needs all feature positions, including absent ones.
        overflow_possible=.false.;largest=maxval(abs(canonical%data(left:right)))
        if(self%max_degree_>3.and.largest>1) then
          overflow_possible=real(self%max_degree_,dp)*log(largest)>=log(huge(1._dp))*(1-32*epsilon(1._dp))
        end if
        if(overflow_possible) then
          m=x%n_cols;allocate(columns(m),values(m),stored(m));columns=[(j,j=1,m)];values=0;stored=.false.
          do j=left,right
            values(canonical%indices(j))=canonical%data(j);stored(canonical%indices(j))=.true.
          end do
        else
          columns=canonical%indices(left:right);values=canonical%data(left:right);allocate(stored(m));stored=.true.
        end if
        allocate(power(m));degree=max(1_int64,self%min_degree_);last=self%max_degree_
        if(self%interaction_only) last=min(last,int(m,int64))
        do while(degree<=last)
          offset=output_count(int(x%n_cols,int64),self%min_degree_,degree-1,self%interaction_only,self%include_bias)
          call initial_power(power,degree,self%interaction_only)
          do
            value=1
            do j=1,m
              repetition=0
              do while(repetition<power(j))
                repetition=repetition+1;value=ieee_product(value,values(j))
              end do
            end do
            keep=self%max_degree_<=3.or.degree==1
            if(degree==1.and.overflow_possible) then
              do j=1,m
                if(power(j)==1) then
                  keep=stored(j);exit
                end if
              end do
            end if
            if(.not.keep) then
              if(degree/=1) then
                keep=ieee_is_nan(value);if(.not.keep) keep=value/=0
              end if
            end if
            if(keep) then
              column=column_rank(columns,power,int(x%n_cols,int64),degree,self%interaction_only)+offset+1
              call require(column>=1.and.column<=self%n_output_features_,'polynomial internal CSR column failure')
              call push(indices,data,used,column,value)
            end if
            call next_power(power,self%interaction_only,done);if(done) exit
          end do
          if(degree==last) exit
          degree=degree+1
        end do
        deallocate(power,columns,values,stored)
      end if
      z%indptr(row+1)=used+1
    end do
    z%indices=indices(:used);z%data=data(:used);call z%validate()
  end function
  function poly_powers(self) result(powers)
    class(polynomial_features), intent(in) :: self
    integer(int64), allocatable :: powers(:,:),power(:)
    integer(int64) :: degree,last,row
    integer :: status
    logical :: done
    call fitted(self)
    call require(self%n_output_features_<=huge(1_int64)/8/self%n_features_in_, &
      'polynomial powers allocation exceeds addressable range')
    allocate(powers(self%n_output_features_,self%n_features_in_),power(self%n_features_in_),stat=status)
    call require(status==0,'polynomial powers allocation failed');row=1
    if(self%include_bias) then
      powers(1,:)=0;row=2
    end if
    degree=max(1_int64,self%min_degree_);last=self%max_degree_
    if(self%interaction_only) last=min(last,int(self%n_features_in_,int64))
    do while(degree<=last)
      call initial_power(power,degree,self%interaction_only)
      do
        powers(row,:)=power;row=row+1
        call next_power(power,self%interaction_only,done);if(done) exit
      end do
      if(degree==last) exit
      degree=degree+1
    end do
  end function
  function poly_names(self,input_features) result(names)
    class(polynomial_features), intent(in) :: self
    type(feature_string), optional, intent(in) :: input_features(:)
    type(feature_string), allocatable :: names(:),input(:)
    integer(int64), allocatable :: power(:)
    integer(int64) :: degree,last,row,length
    integer :: i,pos,status
    character(32) :: number
    logical :: done,first_term
    call fitted(self)
    if(present(input_features)) then
      call require(size(input_features)==self%n_features_in_,'PolynomialFeatures input name count mismatch')
      do i=1,size(input_features)
        call require(allocated(input_features(i)%text),'PolynomialFeatures feature name is unallocated')
        if(allocated(self%feature_names_in_)) then
          call require(len(input_features(i)%text)==len(self%feature_names_in_(i)%text), &
            'PolynomialFeatures input feature names differ')
          call require(input_features(i)%text==self%feature_names_in_(i)%text,'PolynomialFeatures input feature names differ')
        end if
      end do
      input=input_features
    else if(allocated(self%feature_names_in_)) then
      input=self%feature_names_in_
    else
      allocate(input(self%n_features_in_))
      do i=1,size(input)
        number=decimal_text(int(i-1,int64));input(i)%text='x'//trim(number)
      end do
    end if
    call require(self%n_output_features_<=huge(1_int64)/max(1,storage_size(input(1))/8), &
      'polynomial feature names allocation exceeds addressable range')
    allocate(names(self%n_output_features_),power(self%n_features_in_),stat=status)
    call require(status==0,'polynomial feature names allocation failed');row=1
    if(self%include_bias) then
      names(1)%text='1';row=2
    end if
    degree=max(1_int64,self%min_degree_);last=self%max_degree_
    if(self%interaction_only) last=min(last,int(self%n_features_in_,int64))
    do while(degree<=last)
      call initial_power(power,degree,self%interaction_only)
      do
        length=0
        do i=1,size(power)
          if(power(i)==0) cycle
          length=add_count(length,int(len(input(i)%text),int64)+1)
          if(power(i)>1) then
            number=decimal_text(power(i));length=add_count(length,int(len_trim(number)+1,int64))
          end if
        end do
        length=max(0_int64,length-1);call require(length<huge(1),'polynomial feature name exceeds native character range')
        allocate(character(length)::names(row)%text);pos=1;first_term=.true.
        do i=1,size(power)
          if(power(i)==0) cycle
          if(.not.first_term) then
            names(row)%text(pos:pos)=' ';pos=pos+1
          end if
          first_term=.false.;names(row)%text(pos:pos+len(input(i)%text)-1)=input(i)%text;pos=pos+len(input(i)%text)
          if(power(i)>1) then
            number=decimal_text(power(i))
            names(row)%text(pos:pos+len_trim(number))='^'//trim(number);pos=pos+len_trim(number)+1
          end if
        end do
        row=row+1;call next_power(power,self%interaction_only,done);if(done) exit
      end do
      if(degree==last) exit
      degree=degree+1
    end do
  contains
    function decimal_text(value) result(text)
      integer(int64), intent(in) :: value
      character(32) :: text,buffer
      integer(int64) :: remaining
      integer :: at
      ! Nonnegative indices/exponents need no formatted-I/O state or allocation.
      remaining=value;at=len(buffer)
      do
        buffer(at:at)=achar(iachar('0')+int(modulo(remaining,10_int64)))
        remaining=remaining/10;at=at-1
        if(remaining==0) exit
      end do
      text=buffer(at+1:)
    end function
  end function
  function poly_fit_transform(self,x,feature_names) result(z)
    class(polynomial_features), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,feature_names);z=self%transform(x)
  end function
  function poly_fit_transform_sparse(self,x,feature_names) result(z)
    class(polynomial_features), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(feature_string), optional, intent(in) :: feature_names(:)
    type(csr_matrix) :: z
    call self%fit(x,feature_names);z=self%transform(x)
  end function
  function polynomial_array(x,degree,include_bias,interaction_only,min_degree) result(z)
    real(dp), intent(in) :: x(:,:)
    integer, optional, intent(in) :: degree,min_degree
    logical, optional, intent(in) :: include_bias,interaction_only
    real(dp), allocatable :: z(:,:)
    type(polynomial_features) :: model
    if(present(degree)) model%degree=degree
    if(present(min_degree)) model%min_degree=min_degree
    if(present(include_bias)) model%include_bias=include_bias
    if(present(interaction_only)) model%interaction_only=interaction_only
    z=model%fit_transform(x)
  end function
  logical function bridge_fitted(self) result(ok)
    class(polynomial_features), intent(in) :: self
    ok=self%n_features_in_>0
  end function
  integer function bridge_features(self) result(value)
    class(polynomial_features), intent(in) :: self
    value=self%n_features_in_
  end function
end module
