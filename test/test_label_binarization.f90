program test_label_binarization
  use sklearn_label_binarization, only: label_binarizer,multi_label_binarizer,label_set
  use sklearn_common, only: dp,require
  implicit none
  type(label_binarizer) :: binary
  type(multi_label_binarizer) :: multi
  type(label_set) :: sets(4)
  type(label_set), allocatable :: restored(:)
  integer, allocatable :: z(:,:),labels(:)
  integer :: y(4)=[7,-3,7,-3]
  z=binary%fit_transform(y)
  call require(all(binary%classes_==[-3,7]),'LabelBinarizer sorted classes')
  call require(all(shape(z)==[4,1]).and.all(z(:,1)==[1,0,1,0]),'LabelBinarizer positive class column')
  labels=binary%inverse_transform(z)
  call require(all(labels==y),'LabelBinarizer integer round trip')
  labels=binary%inverse_transform(reshape([.5_dp,.6_dp,.1_dp,.9_dp],[4,1]))
  call require(all(labels==[-3,7,-3,7]),'LabelBinarizer strict midpoint threshold')
  z=binary%transform([-3,7,99])
  call require(all(shape(z)==[3,2]).and.all(z(3,:)==0),'LabelBinarizer multiclass query width')
  binary%neg_label=-4; binary%pos_label=0; z=binary%transform(y)
  call require(all(z(:,1)==[0,-4,0,-4]),'LabelBinarizer live labels and zero positive value')
  call binary%fit([9,1,5,9])
  labels=binary%inverse_transform(reshape([1._dp,1._dp,0._dp,0._dp,1._dp,0._dp],[2,3]))
  call require(all(labels==[1,1]),'LabelBinarizer multiclass first maximum tie')
  labels=binary%inverse_transform(reshape([0._dp,0._dp,0._dp,1._dp],[1,4]))
  call require(all(labels==[9]),'LabelBinarizer source clipped score index')
  call binary%fit([7,7]); z=binary%transform([7,99])
  call require(all(z==-4),'LabelBinarizer one-class constant negative encoding')
  labels=binary%inverse_transform(reshape([-4._dp,-4._dp],[1,2]))
  call require(all(labels==[7]),'LabelBinarizer one-class two-column negative scores')
  labels=binary%inverse_transform(z(:0,:))
  call require(size(labels)==0,'LabelBinarizer empty inverse rows')
  sets(1)%values=[7,-3,7]; sets(2)%values=[2]; sets(4)%values=[-3,2]
  z=multi%fit_transform(sets)
  call require(all(multi%classes_==[-3,2,7]),'MultiLabelBinarizer sorted union')
  call require(all(z(1,:)==[1,0,1]).and.all(z(3,:)==0),'MultiLabelBinarizer duplicates and empty set')
  restored=multi%inverse_transform(z)
  call require(all(restored(1)%values==[-3,7]).and.size(restored(3)%values)==0, &
    'MultiLabelBinarizer ordered inverse sets')
  multi%classes=[7,-3]; z=multi%fit_transform(sets)
  call require(all(multi%classes_==[7,-3]).and.all(z(2,:)==0),'MultiLabelBinarizer custom order and unknown labels')
  multi%classes=[2,7] ! Changing constructor classes waits until the next fit.
  z=multi%transform(sets)
  call require(all(multi%classes_==[7,-3]),'MultiLabelBinarizer fitted class snapshot')
  restored=multi%inverse_transform(real(z,dp))
  call require(all(restored(1)%values==[7,-3]),'MultiLabelBinarizer real indicator inverse')
  deallocate(multi%classes); call multi%fit(sets(:0)); z=multi%transform(sets)
  call require(all(shape(z)==[4,0]),'MultiLabelBinarizer empty fitted vocabulary')
  restored=multi%inverse_transform(z)
  call require(size(restored)==4.and.size(restored(1)%values)==0,'MultiLabelBinarizer zero-column inverse')
  call require(all(sets(1)%values==[7,-3,7]),'Binarizers preserve input sets')
  print *, 'Label binarization tests passed'
end program
